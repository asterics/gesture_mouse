import numpy as np


def clamp(value, min_val, max_val):
    return min(max(value, min_val), max_val)


def clamp_np(value, min_val, max_val):
    return np.minimum(np.maximum(value, min_val), max_val)
