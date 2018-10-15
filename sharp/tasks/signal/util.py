"""
Utility functions for signal sample indices, times, and the conversion between
them.
"""

import numpy as np
from numpy.core.multiarray import ndarray

from sharp.data.types.aliases import ArrayLike, IndexList


def time_to_index(
    t: ArrayLike, fs: float, arr_size: int = np.inf, clip: bool = False
) -> IndexList:
    """
    Convert times to array indices.

    :param t:  Times, in seconds
    :param fs:  Sampling frequency, in hertz.
    :param arr_size:  Size of the array in which the indices will be used.
    :param clip:  If True, clips the indices between 0 and the `arr_size`.
                If False (default), raises a ValueError when the indices cannot
                be used to index an array of size `arr_size`.
    :return: Indices.
    """
    indices = (np.array(t) * fs).round()
    if clip:
        return indices.clip(0, arr_size - 1).astype("int")
    else:
        if np.all(indices >= 0) and np.all(indices < arr_size):
            return indices.astype("int")
        else:
            raise ValueError(
                f"Times {t} cannot be used to index an array of size "
                f"{arr_size} at sampling frequency {fs}."
            )


def view(time: ArrayLike, *args) -> ndarray:
    """
    A range around some timestamp(s). Useful for plotting.

    :param time
    :param args:  either not given, or one of {dt, (t1, t2)}
    :return:  Vector of length 2 when input `t` is scalar.
             N x 2 array when input `t` is a vector of length N.

    Usage:
        view(t)          -->  (t-0.1, t+0.1)
        view(t, dt)      -->  (t-dt, t+dt)
        view(t, t1, t2)  -->  (t-t1, t+t2)
    """
    if len(args) == 0:
        t_before = t_after = 0.1
    elif len(args) == 1:
        t_before, = args
        t_after = t_before
    else:
        t_before, t_after = args

    around = np.array([-t_before, +t_after])

    if hasattr(time, "__iter__"):
        return np.array(time)[:, np.newaxis] + around[np.newaxis, :]
    else:
        return time + around
