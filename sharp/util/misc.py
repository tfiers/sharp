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
