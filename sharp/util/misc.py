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


def format_duration(seconds: float, ms_digits: int = 1) -> str:
    """
    A length of time, in human-readable format.
    
    Examples:
        "1h22 43.6s"
        "0h00 05.0s"
        "21.512 ms"
    """
    if seconds < 0:
        return f"{seconds * 1000:.3f} ms"
    else:
        minutes, seconds = divmod(seconds, 60)
        minutes = int(minutes)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:g}h{minutes:02d} {seconds:02d.{ms_digits}f}s"
