import numpy as np


def grayscale(arr):
    return np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
