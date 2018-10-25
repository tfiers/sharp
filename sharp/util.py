from contextlib import contextmanager
from datetime import date
from functools import lru_cache
from logging import getLogger
from logging.config import fileConfig
from shutil import rmtree
from typing import Iterable, Type, Union
from warnings import catch_warnings, simplefilter

import numba
from luigi import Task
from luigi.interface import core
from luigi.task import flatten

from sharp.data.files.base import FileTarget
from sharp.config.params import data_config, output_root

log = getLogger(__name__)


def clear_all_output():
    if output_root.exists():
        # Ignore the sporadic (and wrong)
        # "OSError: [WinError 145] The directory is not empty: ..."
        rmtree(output_root, ignore_errors=True)
        log.info(f"Cleared directory {output_root}.")


def clear_output(task: Task):
    """
    Recursively expands tasks without outputs until it finds tasks that output
    one or more FileTargets. Then deletes these FileTargets.
    """
    if not task.output():
        for dependency in flatten(task.requires()):
            clear_output(dependency)
    else:
        _clear(task.output())


def _clear(target: Union[FileTarget, Iterable[FileTarget]]):
    if isinstance(target, FileTarget):
        target.delete()
    else:
        try:
            for t in target:
                _clear(t)
        except TypeError:
            return


def init_log():
    # Luigi hasn't initalised logging yet when this is called in __main__.py,
    # so we apply the config file ourselves.
    fileConfig(core().logging_conf_file, disable_existing_loggers=False)
    log = getLogger("sharp")
    log.info(f"Hello, it's {date.today():%b %d, %Y}.")
    log.info(f"Output root directory: {output_root}")
    log.info(f"Data config: {data_config}")


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


# Short alias. (We don't care that it's an _LRU_ cache, and we don't want to
# change the cache size).
# NOTE: when combined with @property, @cached goes nearest to the function.
cached = lru_cache(maxsize=256)

# Another short alias
compiled = numba.jit(cache=True, nopython=True)
