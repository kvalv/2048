from typing import List
from enum import Enum

import numpy as np
import gym


class EnumSpace(gym.spaces.Discrete):
    """gym.Space wrapper for an Enum.

    We need to inherit from gym.spaces.Discrete (or some other appropriate spaces), because
    otherwise ray.rllib will complain (with a NotImplementedError).
    """

    def __init__(self, enum):
        self.enum = enum
        self.size = len(self.enum.__members__)
        self.shape = (self.size,)
        self.n = self.size

    def sample(self) -> Enum:
        idx = np.random.randint(0, self.size) + 1
        return self.enum(idx)  # 1-indexed, so we add 1

    def contains(self, x):
        return x in self.enum

    def to_jsonable(self, sample_n):
        return sample_n

    def from_jsonable(self, sample_n):
        return sample_n
