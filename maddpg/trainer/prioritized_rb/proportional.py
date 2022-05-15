import numpy as np
import random
import copy
from . import sum_tree


class Experience(object):
    """ The class represents prioritized experience replay buffer.
    The class has functions: store samples, pick samples with
    probability in proportion to sample's priority, update
    each sample's priority, reset alpha.
    see https://arxiv.org/pdf/1511.05952.pdf .
    """

    def __init__(self, memory_size, batch_size, alpha):
        """ Prioritized experience replay buffer initialization.

        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.tree = sum_tree.SumTree(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha

    def add(self, data, priority):
        """ Add new sample.

        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        self.tree.add(data, priority ** self.alpha)

    def n_step(self, n, r_position, x_position, gamma,num_actor_workers):
        current_index = self.tree.cursor - 1
        current_value = self.tree.data[current_index][r_position]   # reward
        begin_index = current_index
        for i in range(1, n, 1):
            index = begin_index - i*num_actor_workers
            if index < 0 and index + self.tree.filled_size() <= current_index:
                break
            i_gamma = np.power(gamma, i)
            self.tree.data[index][r_position] += i_gamma * current_value   # n-step 的处理,加了4个真实的reward
        if self.tree.filled_size() >= n:
            n_step_back = current_index - n
            if n_step_back < 0 and n_step_back + self.tree.filled_size() <= current_index:
                return
            self.tree.data[n_step_back][x_position] = copy.deepcopy(self.tree.data[current_index][x_position])

    def select(self, beta,num_actor_workers,rnn_length):
        """ The method return samples randomly.

        Parameters
        ----------
        beta : float

        Returns
        -------
        out :
            list of samples
        weights:
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """

        if self.tree.filled_size() < self.batch_size:
            return None, None, None
        ranges = np.linspace(0, self.tree.tree[0], num=self.batch_size + 1)
        out = []
        indices = []
        weights = []
        priorities = []
        for i in range(self.batch_size):
            while True:
                r = random.uniform(ranges[i], ranges[i+1])
                data, priority, index = self.tree.find(r, norm=False)
                if index < (rnn_length-1)*num_actor_workers:
                    index += (rnn_length-1)*num_actor_workers
                    data=self.tree.data[index]
                    priority=self.tree.tree[index + (2 ** (self.tree.tree_level - 1) - 1)]
                if data is not None:
                    break
            priorities.append(priority)
            weights.append((1. / self.memory_size / priority) ** beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)

        weights = list(np.array(weights) / max(weights))  # Normalize for stability

        return out, weights,priorities, indices

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p ** self.alpha)

    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.
        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i) ** -old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)




