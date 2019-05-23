"""
Note: this module sits at the top of the import graph for sharp, i.e we
should'nt import from other sharp modules here.
"""
from contextlib import contextmanager
from functools import lru_cache
from typing import Type
from warnings import catch_warnings, simplefilter

import numba


# Short alias. (We don't care that it's an _LRU_ cache, and we don't want to
# change the cache size).
# NOTE: when combined with @property, @cached goes nearest to the function.
cached = lru_cache(maxsize=256)

# Another short alias
compiled = numba.jit(cache=True, nopython=True)


@contextmanager
def ignore(warning_type: Type[Warning]):
    """
    Executes a block of code while ignoring certain warnings. Usage example:

        >>> with ignore(FutureWarning):
        ...     scipy.filtfilt(b, a, signal)

    """
    with catch_warnings():
        simplefilter("ignore", warning_type)
        yield


def format_duration(
    duration_in_seconds: float, auto_ms: bool = False, ms_digits: int = 1
) -> str:
    """
    
    :param duration_in_seconds
    :param auto_ms:  Whether to switch to "SS.MMM ms" format when the given
            duration is shorter than a second.
    :param ms_digits:  Number of digits for the millisecond part in the "1h22m
            43.784s" format.
    :return:  The given length of time in human-readable format.
    
    Examples:
        "1h22 43.6s"
        "0h00 05.0s"
        "21.512 ms"
    """
    if auto_ms and duration_in_seconds < 1:
        return f"{duration_in_seconds * 1000:.3f} ms"
    else:
        # Explicit conversion to int, from float64 e.g.
        # (int is needed for "02d" format string).
        duration_in_ms = int(round(duration_in_seconds * 1000))
        seconds, milliseconds = divmod(duration_in_ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:g}h{minutes:02d}m {seconds:02d}.{milliseconds:0{ms_digits}d}s"
