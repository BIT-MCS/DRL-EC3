import numpy as np
import random
from .proportional import Experience

class ReplayBuffer(object):
    def __init__(self, size, batch_size, alpha, epsilon):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.rb = Experience(size, batch_size, alpha)
        self.epsilon = epsilon
        # self._storage = []
        # self._maxsize = int(size)
        # self._next_idx = 0

    def __len__(self):
        # return len(self._storage)
        return self.rb.tree.filled_size()

    def clear(self):
        # self._storage = []
        # self._next_idx = 0
        self.rb = Experience(self.rb.memory_size, self.rb.batch_size, self.rb.alpha)

    def add(self, obs_t, action, reward, obs_tp1, done, n, gamma,num_actor_workers):
        data = [obs_t, action, reward, obs_tp1, done]
        priority = self.rb.tree.max_value + self.epsilon
        self.rb.add(data, priority)
        reward_index = 2   # reward的位置是2
        x__index = 3 # next_state的位置是3

        # TODO：刘老师想在这里做文章~~
        self.rb.n_step(n, reward_index, x__index, gamma,num_actor_workers)

    def _encode_sample(self, data):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in data:
            # data = self.rb.tree.data[i]
            obs_t, action, reward, obs_tp1, done = i
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _encode_sample_index(self, index,num_actor_workers,rnn_length):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in index:
            if rnn_length>0:
                obs_t_rnn, action_rnn, reward_rnn, obs_tp1_rnn, done_rnn=[],[],[],[],[]
                for j in range(rnn_length-1,-1,-1):
                    pindex=i-j*num_actor_workers
                    obs_t, action, reward, obs_tp1, done = self.rb.tree.data[pindex]

                    obs_t_rnn.append(np.array(obs_t))
                    action_rnn.append(np.array(action))
                    reward_rnn.append(np.array(reward))
                    obs_tp1_rnn.append(np.array(obs_tp1))
                    done_rnn.append(np.array(done))
                obses_t.append(np.array(obs_t_rnn, copy=False))
                actions.append(np.array(action_rnn, copy=False))
                rewards.append(reward_rnn)
                obses_tp1.append(np.array(obs_tp1_rnn, copy=False))
                dones.append(done_rnn)

            else:
                obs_t, action, reward, obs_tp1, done = self.rb.tree.data[i]
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    # def make_index(self, batch_size):
    #     return [random.randint(0, self.rb.tree.filled_size() - 1) for _ in range(batch_size)]
    #
    # def make_latest_index(self, batch_size):
    #     idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
    #     np.random.shuffle(idx)
    #     return idx

    def sample_index(self, idxes,num_actor_workers,rnn_length):
        return self._encode_sample_index(idxes,num_actor_workers,rnn_length)

    def sample(self, batch_size, beta,num_actor_workers,rnn_length):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        data, weight,priorities, indices = self.rb.select(beta,num_actor_workers,rnn_length)
        return self._encode_sample(data), weight,priorities, indices

    def priority_update(self, indices, priorities):
        priorities = list(np.array(priorities) + self.epsilon)
        self.rb.priority_update(indices=indices, priorities=priorities)

    def reset_alpha(self, alpha):
        self.rb.reset_alpha(alpha)

    # def collect(self):
    #     return self.sample(-1)
