# src/bee-mic/util.py
"""
A few utility functions

Try not to dump things here

"""

import numpy as np


def pad_sum(*arrays, axis=-1):
    """
    Zero-pads multiple NumPy arrays along `axis` to match the longest array,
    then returns their element-wise sum.

    Useful for mixing simulated multi-channel recordings (num_mics, num_samples)
    that have slightly different propagation tail lengths.
    """
    if not arrays:
        raise ValueError("At least one array must be provided to pad_sum.")

    arr_list = [np.asarray(a) for a in arrays]
    max_len = max(a.shape[axis] for a in arr_list)

    padded = []
    for a in arr_list:
        diff = max_len - a.shape[axis]
        if diff > 0:
            pad_width = [(0, 0)] * a.ndim
            pad_width[axis] = (0, diff)
            a = np.pad(a, pad_width, mode='constant', constant_values=0)
        padded.append(a)

    return np.sum(padded, axis=0)