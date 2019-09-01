import gym
import numpy as np


class BoardSpace(gym.Space):
    """A `gym.Space` wrapper for a 2048 board"""

    def __init__(self, n_rows, n_cols):
        self.shape = (n_rows, n_cols)

    def sample(self):
        # shouldn't be necessary
        raise NotImplementedError

    def contains(self, x):
        values = x.as_array()
        return np.all(values >= 0)

    def to_jsonable(self, sample_n):
        raise NotImplementedError

    def from_jsonable(self, sample_n):
        raise NotImplementedError
