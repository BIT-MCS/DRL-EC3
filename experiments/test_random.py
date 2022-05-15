import argparse
import numpy as np
import os
import tensorflow as tf
import pandas as pd

import maddpg.common.tf_util as U
from experiments.env0 import log0 as Log
from experiments.env0.data_collection0 import Env
from maddpg.trainer.maddpg import MADDPGAgentTrainer


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Hyperparameters
ARGUMENTS = [
    # Environment
    ["--scenario", str, "simple_adversary", "name of the scenario script"],
    ["--max-episode-len", int, 500, "maximum episode length"],
    ["--num-episodes", int, 5000, "number of episodes"],
    ["--num-adversaries", int, 0, "number of adversaries(enemy)"],
    ["--good-policy", str, "maddpg", "policy for good agents"],
    ["--adv-policy", str, "maddpg", "policy of adversaries"],

    # Core training parameters
    ["--lr", float, 5e-4, "learning rate for Adam optimizer"],
    ["--decay_rate", float, 0.99995, "learning rate exponential decay"], # 作为初始学习率，后面尝试进行衰减,这个不着急加!
    ["--gamma", float, 0.95, "discount factor"],
    ["--batch-size", int, 512, "number of epochs to optimize at the same time"],
    ["--num-units", int, 600, "number of units in the mlp"],

    # Priority Replay Buffer ( weights not used )
    ["--alpha", float, 0.5, "priority parameter"],
    ["--beta", float, 0.4, "IS parameter"],
    ["--epsilon", float, 0.5, "a small positive constant"],
    ["--buffer_size", int, 200000, "buffer size for each agent"],

    # N-steps
    ["--N", int, 5, "steps of N-step"],

    # TODO: Experiments
    # Ape-X
    ["--num_actor_workers", int,0,
     "number of environments one agent can deal with. if >1, use apex ; else, use simple maddpg"],
    ["--debug_dir", str, "/debug_list/",
     "save index,reward(n-step),priority, value,wi per every sample from experience"],

    # RNN
    ["--rnn_length", int,0,
     "time_step in rnn, try to use LSTM instead of N-steps. if ==0, not use rnn; else, use rnn."],
    ["--rnn_cell_size", int, 64, "LSTM-cell output's size"],

    # Checkpointing 保存model
    ["--exp-name", str, None, "name of the experiment"],
    ["--save-dir", str, "/policy/", "directory in which training state and model sho uld be saved"],
    ["--save-rate", int, 10, "save model once every time this many episodes are completed"],
    ["--model_to_keep", int, 100, "the number of saved models"],
    ["--load-dir", str, "/media/sda1/MCS_experiments/test_裸奔/uav5",
     "directory in which training state and model are loaded"],

    # Test
    ['--test_time', int, 10, "number of iterations run for testing"],
    ["--random_seed", int, 100, "random seed"],
    ["--start", int,0,"start model"],
    ["--end", int,5, "end model"]
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


def test(arglist, log,full_load_dir,test_iteration):

    with U.multi_threaded_session() as sess:
        # Create environment for testing
        env=Env(log)
        log.log(ARGUMENTS)
        log.log(ACTIONS)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)  # 0
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)  # 定义所有数据结构和静态图

        # Initialize all the uninitialized variables in the global scope
        U.initialize()

        # TODO:加载已经训练好的模型
        saver = tf.train.Saver()
        saver.restore(sess,full_load_dir)

        episode_rewards = [0.0]   # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        episode_step = 0

        start_env=env.reset()
        state_step = []
        for _ in range(0, arglist.rnn_length - 1):
            state_step.append(start_env)
        obs_n=start_env


        print('Starting a new TEST iterations...')
        print('Log_dir:', env.log_dir)
        iteration = 0

        efficiency=[]
        fairness=[]
        normal_fairness=[]
        collection_ratio=[]
        energy_consumption=[]
        collision = []
        steps = []

        indicator = [0] * env.n  # TODO:状态指示器
        meaningful_fill = [0] * env.n
        meaningful_get = [0] * env.n

        # testing
        while iteration < arglist.test_time:
            if arglist.rnn_length > 0:
                action_n = []
                state_step.append(obs_n)
                for i, agent, obs in zip(range(0, len(trainers)), trainers, obs_n):
                    obs_sequence = []
                    for j in range(-1 * arglist.rnn_length, 0, 1):
                        obs_sequence.append(state_step[j][i])

                    action_n.append(agent.action(np.array(obs_sequence)))
            else:
                action_n = [agent.action(obs[None]) for agent, obs in zip(trainers, obs_n)]

            action_n=np.array(action_n)
            random_action_n=np.random.uniform(low=-1,high=1,size=action_n.shape)
            new_obs_n, rew_n, done_n, info_n, indicator = env.step(actions=random_action_n, indicator=indicator)
            log.step_information(random_action_n, env, episode_step, iteration, meaningful_fill, meaningful_get,
                                 indicator)

            indicator = [0] * env.n
            episode_step += 1  # step per episode
            done = done_n
            terminal = (episode_step >= arglist.max_episode_len)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew  # 每一个step的总reward
                agent_rewards[i][-1] += rew  # 每一个step,每个agent自己的reward

            if done or terminal:
                efficiency.append(env.efficiency)
                fairness.append(env.collection_fairness)
                normal_fairness.append(env.normal_collection_fairness)
                collection_ratio.append(1.0-env.leftrewards)
                energy_consumption.append(np.sum(env.normal_use_energy))
                collision.append(np.sum(env.walls))
                steps.append(env.count)

                log.draw_path(env, iteration, meaningful_fill, meaningful_get)

                iteration += 1
                meaningful_fill = [0] * env.n
                meaningful_get = [0] * env.n
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)

            # for displaying learned policies
            if arglist.display:
                env.render()
                continue

        details = [
            '\n\nindicator DETAILS:',
            '\n\tefficiency: ' + str(efficiency),
            '\n\tfairness: ' + str(fairness),
            '\n\tnormal_fairness: ' + str(normal_fairness),
            '\n\tcollection_ratio: ' + str(collection_ratio),
            '\n\tenergy_consumption: ' + str(energy_consumption),
            '\n\tcollision: ' + str(collision),
            '\n\tsteps: ' + str(steps),
        ]

        indicator = [
            '\n\ntest_model: '+str(test_iteration)+' --indicator AVERAGE:',
            '\n\tefficiency: ' + str(np.mean(efficiency)),
            '\n\tfairness: ' + str(np.mean(fairness)),
            '\n\tnormal_fairness: ' + str(np.mean(normal_fairness)),
            '\n\tcollection_ratio: ' + str(np.mean(collection_ratio)),
            '\n\tenergy_consumption: ' + str(np.mean(energy_consumption)),
            '\n\tcollision: ' + str(np.mean(collision)),
            '\n\tsteps: ' + str(np.mean(steps)),
        ]

        for _ in indicator:
            print(_)

        indicator_to_pandas = [
            str(test_iteration),

            str(np.mean(collection_ratio)),
            str(np.min(collection_ratio)),
            str(np.max(collection_ratio)),

            str(np.mean(normal_fairness)),
            str(np.min(normal_fairness)),
            str(np.max(normal_fairness)),

            str(np.mean(energy_consumption)),
            str(np.min(energy_consumption)),
            str(np.max(energy_consumption)),

            str(np.mean(efficiency)),
            str(np.min(efficiency)),
            str(np.max(efficiency)),
        ]


        log.log(details)
        log.log(indicator)

    tf.reset_default_graph()
    return indicator_to_pandas

if __name__ == '__main__':
    print('Loading the trained model...Now, Enjoy yourself!')
    arglist = parse_args()
    df=pd.DataFrame(columns=["test_model",
                             "collection_ratio","cr_min","cr_max",
                             "fairness","f_min","f_max",
                             "consumption of energy","ce_min","ce_max",
                             "efficiency","e_min","e_max"])
    for i in range(arglist.start,arglist.end):
        full_load_dir=arglist.load_dir+"/policy/"+str(i)+".ckpt"
        log = Log.Log()
        indicator_to_pandas=test(arglist, log,full_load_dir,i)
        df.loc[i-70]=indicator_to_pandas

    df.sort_values("efficiency",inplace=True)
    df.to_csv(arglist.load_dir+"/瞎跑uav5.csv",index=0)
    print('\n', 'TEST finished')
