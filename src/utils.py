import numpy as np


def random_choice_along_axis(arr, axis):
    idx = np.random.randint(0, arr.shape[axis])
    return np.take(arr, idx, axis)
