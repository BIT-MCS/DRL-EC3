import argparse
import numpy as np
import os
import tensorflow as tf
import time

import maddpg.common.tf_util as U
from experiments.env0 import log0 as Log
from experiments.env0.data_collection0 import Env
from maddpg.common.summary import Summary
from maddpg.trainer.maddpg import MADDPGAgentTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Hyperparameters
ARGUMENTS = [
    # Environment
    ["--scenario", str, "simple_adversary", "name of the scenario script"],
    ["--max-episode-len", int, 500, "maximum episode length"],
    ["--num-episodes", int, 500, "number of episodes"],
    ["--num-adversaries", int, 0, "number of adversaries(enemy)"],
    ["--good-policy", str, "maddpg", "policy for good agents"],
    ["--adv-policy", str, "maddpg", "policy of adversaries"],

    # Core training parameters
    ["--lr", float, 5e-4, "learning rate for Adam optimizer"],
    ["--decay_rate", float, 0.99995, "learning rate exponential decay"], # 作为初始学习率，后面尝试进行衰减,这个不着急加!
    ["--gamma", float, 0.95, "discount factor"],
    ["--batch-size", int, 32, "number of epochs to optimize at the same time"],  # 512
    ["--num-units", int, 600, "number of units in the mlp"],

    # Priority Replay Buffer ( weights not used )
    ["--alpha", float, 0.5, "priority parameter"],
    ["--beta", float, 0.4, "IS parameter"],
    ["--epsilon", float, 0.5, "a small positive constant"],
    ["--buffer_size", int, 200000, "buffer size for each agent"] ,

    # N-steps
    ["--N", int, 5, "steps of N-step"],

    # TODO: Experiments
    # Ape-X
    ["--num_actor_workers", int, 0,
     "number of environments one agent can deal with. if >1, use apex ; else, use simple maddpg"],
    ["--debug_dir", str, "/debug_list/",
     "save index,reward(n-step),priority, value,wi per every sample from experience"],

    # RNN
    ["--rnn_length", int, 0,
     "time_step in rnn, try to use LSTM instead of N-steps. if ==0, not use rnn; else, use rnn."],
    ["--rnn_cell_size", int, 64, "LSTM-cell output's size"],

    # Checkpointing 保存model
    ["--exp-name", str, None, "name of the experiment"],
    ["--save-dir", str, "/policy/", "directory in which training state and model should be saved"],
    ["--save-rate", int, 2, "save model once every time this many episodes are completed"],
    ["--model_to_keep", int, 100, "the number of saved models"],
    ["--load-dir", str, "/home/linc/Desktop/maddpg-final/saved_state.ckpt",
     "directory in which training state and model are loaed"],

    # Evaluation
    ["--benchmark-iters", int, 100000, "number of iterations run for benchmarking"],
    ["--benchmark-dir", str, "./benchm", "directory where benchmark data is saved"],
    ["--plots-dir", str, "./learning_curves/", "directory where plot data is saved"],

    # Training
    ["--random_seed", int, 0, "random seed"]
]

ACTIONS = [
    ["--restore", "store_true", False],
    ["--display", "store_true", False],
    ["--benchmark", "store_true", False]

]


# 参数调节器
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    for arg in ARGUMENTS:
        parser.add_argument(arg[0], type=arg[1], default=arg[2], help=arg[3])
    for action in ACTIONS:
        parser.add_argument(action[0], action=action[1], default=action[2])
    return parser.parse_args()


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    # 加入多个trainers
    trainers = []
    trainer = MADDPGAgentTrainer

    # 对手agent个数  0
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))

    # 盟友agent个数  env.n  每一个agent都有一个actor，critic，replay_buffer！！！
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))

    return trainers


def train(arglist, log):

    with U.multi_threaded_session() as sess:
        # Create environment(use Ape-X)
        envs = [Env(log) for _ in range(arglist.num_actor_workers)]
        log.log(ARGUMENTS)
        log.log(ACTIONS)

        # Create summary
        summary = Summary(sess, envs[0].log_dir)
        for i in range(envs[0].n):
            summary.add_variable(tf.Variable(0.), 'reward_%d' % i)
            summary.add_variable(tf.Variable(0.), 'loss_%d' % i)
            summary.add_variable(tf.Variable(0.), 'wall_%d' % i)
            summary.add_variable(tf.Variable(0.), 'energy_%d' % i)
            summary.add_variable(tf.Variable(0.), 'gained_info_%d' % i)
        summary.add_variable(tf.Variable(0.), 'buffer_size')
        summary.add_variable(tf.Variable(0.), 'acc_reward')
        summary.add_variable(tf.Variable(0.), 'leftrewards')
        summary.add_variable(tf.Variable(0.), 'efficiency')
        summary.build()

        # Create agent trainers
        obs_shape_n = [envs[0].observation_space[i].shape for i in range(envs[0].n)]

        # 计算对手个数
        num_adversaries = min(envs[0].n, arglist.num_adversaries)  # 0

        # 定义所有数据结构和静态图
        trainers = get_trainers(envs[0], num_adversaries, obs_shape_n, arglist)

        # # 我方和敌方采用不同策略(适用于多智能体的双方竞争环境)
        # print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize all the uninitialized variables in the global scope
        U.initialize()

        if arglist.restore:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        # 保存模型的个数 100
        saver = tf.train.Saver(max_to_keep=arglist.model_to_keep)

        episode_rewards = [[0.0] for env in envs]  # sum of rewards for all agents
        agent_rewards = [[[0.0] for _ in range(env.n)] for env in envs]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        obs_n = []
        state_step_n = []
        for env in envs:
            start_env = env.reset()
            state_step_i = []
            for _ in range(0, arglist.rnn_length - 1):
                state_step_i.append(start_env)
            state_step_n.append(state_step_i)
            obs_n.append(start_env)
        episode_step = [0 for env in envs]
        t_start = [time.time() for env in envs]

        m_time = t_start.copy()
        print('Starting iterations...')
        print('Log_dir:', envs[0].log_dir)
        iteration = 0
        global_total_step = 0  # episode step
        loss = [0.] * envs[0].n
        model_index = 0
        efficiency = 0
        indicator = [0] * envs[0].n  # TODO:状态指示器
        meaningful_fill = [0] * envs[0].n
        meaningful_get = [0] * envs[0].n

        # training
        while iteration < arglist.num_episodes:
            global_total_step += 1  # sum step id
            terminal_done_0=False
            # TODO:DEBUG
            # print("global-step: ",global_total_step)
            rew_n_master = []
            for env_i, env in enumerate(envs):
                # get action 各取各的
                # TODO:LSTM try
                if arglist.rnn_length > 0:
                    action_n = []
                    state_step_n[env_i].append(obs_n[env_i])
                    for i, agent, obs in zip(range(0, len(trainers)), trainers, obs_n[env_i]):
                        obs_sequence = []

                        for j in range(-1 * arglist.rnn_length, 0, 1):
                            obs_sequence.append(state_step_n[env_i][j][i])

                        action_n.append(agent.action(np.array(obs_sequence)))
                else:
                    action_n = [agent.action(obs[None]) for agent, obs in zip(trainers, obs_n[env_i])]

                # environment step
                if env_i == 0:
                    # TODO:加入状态指示器放在step里面进行每一步的更新
                    new_obs_n, rew_n, done_n, info_n, indicator = env.step(actions=action_n, indicator=indicator)
                    # TODO：step-debug
                    log.step_information(action_n, env, episode_step[0], iteration, meaningful_fill, meaningful_get,
                                         indicator)
                    rew_n_master = rew_n
                    indicator = [0] * envs[0].n
                else:
                    new_obs_n, rew_n, done_n, info_n, _ = env.step(actions=action_n)
                episode_step[env_i] += 1  # step per episode
                done = done_n
                terminal = (episode_step[env_i] >= arglist.max_episode_len)

                # collect experience 添加buffer是各加各的
                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[env_i][i], action_n[i], rew_n[i], new_obs_n[i], done_n, terminal,
                                     arglist.num_actor_workers)
                obs_n[env_i] = new_obs_n

                for i, rew in enumerate(rew_n):
                    episode_rewards[env_i][-1] += rew  # 每一个step的总reward
                    agent_rewards[env_i][i][-1] += rew  # 每一个step,每个agent自己的reward

                if done or terminal:
                    # report
                    if env_i == 0:
                        terminal_done_0=True
                        print('\n%d th episode:\n' % iteration)
                        print('\tthe %d env,%d steps,%.2f seconds, wasted %.2f seconds.' % (
                            env_i, episode_step[env_i], time.time() - m_time[env_i], env.time_))
                        # print('rewards:', agent_rewards[0][-1], agent_rewards[1][-1])
                        print('\tobstacle collisions:', env.walls)
                        print('\tdata collection:', env.collection / env.totaldata)
                        print('\treminding energy:', env.energy)
                        efficiency = env.efficiency
                        # log.draw_path(env, iteration)
                        log.draw_path(env, iteration, meaningful_fill, meaningful_get)
                        iteration += 1

                    meaningful_fill = [0] * envs[0].n
                    meaningful_get = [0] * envs[0].n
                    m_time[env_i] = time.time()
                    obs_n[env_i] = env.reset()
                    episode_step[env_i] = 0
                    episode_rewards[env_i].append(0)
                    for a in agent_rewards[env_i]:
                        a.append(0)
                    agent_info.append([[]])

            # for displaying learned policies
            if arglist.display:
                envs[0].render()
                continue

            # update all trainers, if not in display or benchmark mode
            _loss = []

            # update  每一个agent自己更新自己的PQ参数
            for agent in trainers:  # 将buffer采样初始化
                agent.preupdate()
            for agent in trainers:
                _loss.append(agent.update(envs[0], trainers, global_total_step)[0])
            if np.sum(_loss) != 0:  # 在buffer没有填满的时候不加loss
                loss = _loss

            # summary vistalize for all UAVs
            feed_dict = {}
            for i_summary in range(envs[0].n):
                feed_dict['reward_%d' % i_summary] = rew_n_master[i_summary]
                feed_dict['loss_%d' % i_summary] = loss[i_summary]
                feed_dict['wall_%d' % i_summary] = envs[0].walls[i_summary] / (float(episode_step[0]) + 1e-4)
                feed_dict['energy_%d' % i_summary] = envs[0].energy[i_summary]
                feed_dict['gained_info_%d' % i_summary] = envs[0].collection[i_summary]
            feed_dict['buffer_size'] = trainers[0].filled_size
            feed_dict['leftrewards'] = envs[0].leftrewards
            feed_dict['acc_reward'] = episode_rewards[0][-1]
            feed_dict['efficiency'] = efficiency
            summary.run(feed_dict=feed_dict, step=global_total_step)

            # save model, display training output
            if terminal_done_0 is True and (len(episode_rewards[0]) + 1) % arglist.save_rate == 0:
                U.save_state(
                    envs[0].log_dir + arglist.save_dir + "/" + str(model_index % arglist.model_to_keep) + ".ckpt",
                    saver=saver)
                model_index += 1
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("------------------------------------------------------------------------------------------")
                    print("Master: steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        global_total_step, len(episode_rewards[0]) - 1,
                        np.mean(episode_rewards[0][-arglist.save_rate:]),
                        round(time.time() - t_start[0], 3)))
                    print("------------------------------------------------------------------------------------------")
                else:
                    print("------------------------------------------------------------------------------------------")
                    print(
                        "Master: steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                            global_total_step, len(episode_rewards[0]) - 1,
                            np.mean(episode_rewards[0][-arglist.save_rate:]),
                            [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards],
                            round(time.time() - t_start[0], 3)))
                    print("------------------------------------------------------------------------------------------")
                t_start = [time.time() for env in envs]
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[0][-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))


if __name__ == '__main__':
    print('Let\'s train, go! go! go!')
    arglist = parse_args()
    log = Log.Log()
    train(arglist, log)
