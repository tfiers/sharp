"""
Note: this module sits at the top of the import graph for sharp, i.e we
should'nt import from other sharp modules here.
"""
import re
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Type, Union
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


def make_parent_dirs(file_path: Union[Path, str]):
    """
    Make sure the containing directories exist.
    
    :param file_path:  Pointing to a file in a directory.
    """
    dir_path: Path = Path(file_path).parent
    dir_path.mkdir(exist_ok=True, parents=True)


def linearize(multiline_string: str) -> str:
    """
    :param multiline_string:  ..as found in source code.
    :return:  A string without newlines nor multiple consecutive whitespace
            characters.
    """
    line = re.sub(r"\s+", " ", multiline_string)
    return line.strip()


def format_duration(
    duration_in_seconds: float,
    fixed_len: bool = False,
    precision: int = 1,
    auto_ms: bool = True,
) -> str:
    """
    :return:  The given length of time in human-readable format.
    :param duration_in_seconds:  .
    :param fixed_len:  Whether to output leading 0 parts. See examples below.
    :param precision:  Number of digits after the decimal dot. Must be ≥ 0.
    :param auto_ms:  Whether to use "MM.M ms" format when duration is less than
            a second. Ignored when fixed_len is True.
    
    Output examples for fixed_len = True:
        "1h08m 05.6s"
        "0h08m 05.6s"
        "0h00m 05.6s"
    
    Output examples for fixed_len = False :
        "1h08m 05.6s"
        "8m 05.6s"
        "5.6s"
        "21.5 ms"
    """
    if not fixed_len and auto_ms and duration_in_seconds < 1:
        return f"{duration_in_seconds * 1000:.{precision}f} ms"
    else:
        total_seconds, fractional_part = divmod(duration_in_seconds, 1)
        # Explicit conversion to int, from float or np.float64 e.g.
        # (int is needed for "02d" format string).
        total_seconds = int(total_seconds)
        total_minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(total_minutes, 60)
        decimal_digits = round(fractional_part * 10 ** precision)
        if precision == 0:
            suffix = "s"
        else:
            suffix = f".{decimal_digits:0{precision}d}s"
        if fixed_len or hours > 0:
            return f"{hours}h{minutes:02d}m {seconds:02d}{suffix}"
        elif minutes > 0:
            return f"{minutes}m {seconds:02d}{suffix}"
        else:
            return f"{seconds}{suffix}"
