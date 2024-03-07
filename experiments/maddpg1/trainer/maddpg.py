import os
import numpy as np
import tensorflow as tf
import maddpg1.common.tf_util as U

from maddpg1.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg1.trainer.prioritized_rb.replay_buffer import ReplayBuffer
import tensorflow.contrib.layers as layers


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        # update target network parameters (once)
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def CNN(state_input, reuse=tf.AUTO_REUSE, scope='CNN'):
    with tf.variable_scope(scope, reuse=reuse):
        state = tf.layers.conv2d(state_input, 16, 3, activation='relu', strides=2, padding='VALID')
        state = tf.layers.conv2d(state, 32, 3, activation='relu', strides=2, padding='VALID')
        state = tf.layers.conv2d(state, 64, 3, activation='relu', strides=2, padding='VALID')
        temp = 64 * 9 * 9

        state = tf.layers.batch_normalization(state)
        input_1 = tf.reshape(state, [-1])
        input_s = tf.reshape(input_1, [-1, temp])
    return input_s


# TODO: RNN!!!
def RNN(state_input, reuse=tf.AUTO_REUSE, scope='RNN', cell_size=None, initial_state=None):
    with tf.variable_scope(scope, reuse=reuse):
        rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=cell_size,
                                                         layer_norm=True, norm_gain=1.0, norm_shift=0.0,
                                                         dropout_keep_prob=0.75, dropout_prob_seed=None)
        outputs, final_state = tf.nn.dynamic_rnn(
            cell=rnn_cell, inputs=state_input, initial_state=initial_state, time_major=True, dtype=tf.float32)
        cell_out = outputs[-1, :, :]
    return cell_out, final_state


# 多层感知机 Actor/Critic-Net 互相独立
# TODO:输出加了tanh确实不会梯度爆炸,但是收敛效果变得不是很好
def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, ac_fn=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input

        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu,
                                     weights_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                     biases_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu,
                                     weights_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                     biases_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None,
                                     weights_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                     biases_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
        return out


# actor
def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer,global_step, grad_norm_clipping=None, local_q_func=False,
            num_units=64, scope="trainer", reuse=None, args=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions - DiagGaussian
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # make_pdtype(box(2,))->DiagGaussianPdType(2)

        # set up placeholders
        obs_ph_n = make_obs_ph_n  # n * [None,80,80,3]
        # n * [None, 3]
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]

        # actor output shape(p-shape)：[None,2*3]  2:mean,std  3:one_uav_action_n
        # add cnn for actor here! p_input=CNN(obs_ph_n[p_index])
        if args.rnn_length > 0:
            cnn_output = tf.reshape(CNN(state_input=obs_ph_n[p_index], scope='p_func'),
                                    [args.rnn_length, -1, 64 * 9 * 9])
            p_input, _ = RNN(state_input=cnn_output, scope='p_func', cell_size=args.rnn_cell_size)
        else:
            p_input = CNN(obs_ph_n[p_index], scope='p_func')
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units,
                   ac_fn=tf.nn.tanh)  # TODO:actor的输出加tanh,避免爆炸

        # 提取CNN+BATCH_NORMALIZATION+MLP里面的参数
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)
        #   PdType.pdfromflat(p) => DGPT.pdclass()(p) =>DiagGaussianPd(p)
        #   shape of p [None, 4] => mean:[None, 2] std:[None, 2]

        act_sample = act_pd.sample()  # action == mean + std * tf.random_normal
        mean, logstd = act_pd.mean, act_pd.logstd
        # act_pd.flatparam() === p
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()  # 每个agent更新自己的action,跑多次这个函数最后会形成新的action_input_n

        # add cnn for critic here!
        cnn_obs_ph_n = []
        for obs_ph in obs_ph_n:
            # rnn
            if args.rnn_length > 0:
                cnn_output = tf.reshape(CNN(state_input=obs_ph, scope='q_func'), [args.rnn_length, -1, 64 * 9 * 9])
                cell_out, _ = RNN(state_input=cnn_output, scope='q_func', cell_size=args.rnn_cell_size)
                cnn_obs_ph_n.append(cell_out)
            else:
                cnn_obs_ph_n.append(CNN(state_input=obs_ph, scope='q_func'))

        q_input = tf.concat(cnn_obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([cnn_obs_ph_n[p_index], act_input_n[p_index]], 1)
        # reuse=True, the same critic. 会使用q_func空间里的critic，来更新actor的loss
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars,global_step, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)  # 单个agent的evaluate-actor动作输出函数
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        # add cnn for actor here! p_target_input=CNN(obs_ph_n[p_index])
        if args.rnn_length > 0:
            cnn_output = tf.reshape(CNN(state_input=obs_ph_n[p_index], scope='target_p_func'),
                                    [args.rnn_length, -1, 64 * 9 * 9])
            p_target_input, _ = RNN(state_input=cnn_output,scope='target_p_func',cell_size=args.rnn_cell_size)
        else:
            p_target_input = CNN(obs_ph_n[p_index], scope='target_p_func')

        target_p = p_func(p_target_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",
                          num_units=num_units, ac_fn=tf.nn.tanh)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)  # 更新target network

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)  # 单个agent的target-actor动作输出函数

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}


# critic
def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, global_step, grad_norm_clipping=None, local_q_func=False,
            scope="trainer", reuse=None, num_units=64, args=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions （n * action_n）
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]  # dialog 高斯分布

        # set up placeholders
        obs_ph_n = make_obs_ph_n  # n * [None,80,80,3]

        # add CNN for critic here. cnn_obs_ph_n=CNN(obs_ph_n)
        cnn_obs_ph_n = []  # n * [None,5184]
        for obs_ph in obs_ph_n:
            # rnn
            if args.rnn_length > 0:
                cnn_output = tf.reshape(CNN(state_input=obs_ph, scope='q_func'), [args.rnn_length, -1, 64 * 9 * 9])
                cell_out, _ = RNN(state_input=cnn_output, scope='q_func', cell_size=args.rnn_cell_size)
                cnn_obs_ph_n.append(cell_out)
            else:
                cnn_obs_ph_n.append(CNN(state_input=obs_ph, scope='q_func'))

        # multi-state-placeholder(num_agents)
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        # q_input = n * (s cat a)   shape:[None,n*(5184+3)]
        q_input = tf.concat(cnn_obs_ph_n + act_ph_n, 1)
        if local_q_func:  # false
            q_input = tf.concat([cnn_obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]  # MLP 输出值,增加一维 shape=[None,0]

        # 得到训练所需要的所有(在scope命名空间内)的variable变量（weight/bias/batch_normalization）
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_square = tf.square(q - target_ph)
        q_loss = tf.reduce_mean(q_square)  # Square loss (from batch to one)

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss  # + 1e-3 * q_reg

        # optimizer(Adam)
        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars,global_step,grad_norm_clipping)

        # Create callable functions 建立了一个pipeline,方便后续训练
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=[loss, q_square, q, q_input],
                           updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # add CNN for critic here. tcnn_obs_ph_n=CNN(obs_ph_n)
        tcnn_obs_ph_n = []  # n * [None,5184]  for target
        for obs_ph in obs_ph_n:
            # rnn
            if args.rnn_length > 0:
                cnn_output = tf.reshape(CNN(state_input=obs_ph, scope='target_q_func'),
                                        [args.rnn_length, -1, 64 * 9 * 9])
                cell_out, _ = RNN(state_input=cnn_output, scope='target_q_func', cell_size=args.rnn_cell_size)
                tcnn_obs_ph_n.append(cell_out)
            else:
                tcnn_obs_ph_n.append(CNN(state_input=obs_ph, scope='target_q_func'))

        q_target_input = tf.concat(tcnn_obs_ph_n + act_ph_n, 1)
        if local_q_func:  # false
            q_target_input = tf.concat([tcnn_obs_ph_n[q_index], act_ph_n[q_index]], 1)

        # target network
        target_q = q_func(q_target_input, 1, scope="target_q_func", num_units=num_units)[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGAgentTrainer(AgentTrainer):  # 按照agent_index挨个建立trainer
    def __init__(self, name, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)  # 2
        self.agent_index = agent_index
        self.args = args

        # TODO:加一个自适应学习率衰减(有很多tricks)
        self.global_train_step = tf.Variable(tf.constant(0.0), trainable=False)
        self.decey_lr = tf.train.exponential_decay(learning_rate=self.args.lr, global_step=self.global_train_step,
                                                   decay_steps=100, decay_rate=self.args.decay_rate, staircase=True)
        # multi-state-placeholder(num_agents)
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())

        # Create all the functions necessary to train the model

        # critic
        # q_train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=[loss, q_square, q, q_input],
        #                    updates=[optimize_expr])
        # q_update = make_update_exp(q_func_vars, target_q_func_vars)
        # q_values = U.function(obs_ph_n + act_ph_n, q)
        # target_q_values = U.function(obs_ph_n + act_ph_n, target_q)
        # self.q_debug={'q_values': q_values, 'target_q_values': target_q_values}
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=mlp_model,  # mlp
            optimizer=tf.train.AdamOptimizer(learning_rate=self.decey_lr),  #args.lr
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,  # false
            num_units=args.num_units,  # 600
            args=args,
            global_step=self.global_train_step
        )

        # actor
        # self.act 算的是第agent_index个agent的action_sample！
        # self.p-_debug={'p_values': p_values, 'target_act': target_act}
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=mlp_model,  # mlp
            q_func=mlp_model,  # mlp
            optimizer=tf.train.AdamOptimizer(learning_rate=self.decey_lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,  # false
            num_units=args.num_units,  # 600
            args=args,
            global_step = self.global_train_step
        )

        # Create experience buffer
        self.buffer_size = self.args.buffer_size  # 1e6
        self.beta = self.args.beta
        self.replay_buffer = ReplayBuffer(int(self.buffer_size), int(self.args.batch_size), self.args.alpha,
                                          self.args.epsilon)
        self.replay_sample_index = None

    @property
    def filled_size(self):
        return len(self.replay_buffer)

    def action(self, obs):
        actor_output = self.act(obs)[0]
        return actor_output

    def experience(self, obs, act, rew, new_obs, done, terminal, num_actor_workers):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done), self.args.N, self.args.gamma, num_actor_workers)

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, env, agents, t):
        # replay buffer is not large enough 没填满的时候不训练
        # if len(self.replay_buffer) < 10000:
        if len(self.replay_buffer) < 100 * self.args.batch_size:
            return [0]
        if not t % 10 == 0:  # only update every 10 steps
            return [0]

        # 随着训练的进行，让β从某个小于1的值渐进地靠近1
        if self.beta < 1.:
            self.beta *= 1. + 1e-4

        # sample from one agent(batch:1024)  之后根据β算出来的weights没有用到呢！！！
        (obs, act, rew, obs_next, done), weights, priorities, self.replay_sample_index = self.replay_buffer.sample(
            self.args.batch_size, self.beta, self.args.num_actor_workers, self.args.rnn_length)  # batch-size=1024

        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []

        index = self.replay_sample_index  # index数组
        for i in range(self.n):
            obs_, _, rew_, obs_next_, _ = agents[i].replay_buffer.sample_index(index, self.args.num_actor_workers,
                                                                               self.args.rnn_length)
            _, act_, _, _, done_ = agents[i].replay_buffer.sample_index(index, 0, 0)

            if self.args.rnn_length > 0:
                obs_ = obs_.transpose((1, 0, 2, 3, 4))
                obs_next_ = obs_next_.transpose((1, 0, 2, 3, 4))
                obs_shape = obs_.shape
                obs_ = obs_.reshape(-1, obs_shape[-3], obs_shape[-2], obs_shape[-1])
                obs_next_ = obs_next_.reshape(-1, obs_shape[-3], obs_shape[-2], obs_shape[-1])

            obs_n.append(obs_)
            obs_next_n.append(obs_next_)
            act_n.append(act_)

        # train q network
        num_sample = 1
        target_q = 0.0

        # TODO: 在target network里面采用兼顾过去和未来的一长段RNN 计算Qt+n
        # use functions defined (batch:1024)
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma ** self.args.N * (1.0 - done) * target_q_next  # N-step(N=5)
        target_q /= num_sample

        [q_loss, q_td, Q, q_input] = self.q_train(*(obs_n + act_n + [target_q]))

        debug_dir = env.log_dir + self.args.debug_dir
        if os.path.exists(debug_dir) is False:
            os.makedirs(debug_dir)
        with open(debug_dir + "current_step_information_{}.txt".format(self.name), 'w+') as file:
            for i, r, p, q, w in zip(index, rew, priorities, Q, weights):
                print(self.name, " current_global_step: ", t, "-----index: ", i, " reward(n-step): ", r, " priority: ",
                      p, " Q: ", q, " Wi: ", w, file=file)

        # priority replay buffer update (use TD-error)
        values = np.fabs(q_td)
        self.replay_buffer.priority_update(self.replay_sample_index, values)

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()  # actor update
        self.q_update()  # critic update

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
