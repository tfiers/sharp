from contextlib import contextmanager
from functools import lru_cache
from logging import getLogger
from logging.config import fileConfig
from shutil import rmtree
from typing import Iterable, Type, Union
from warnings import catch_warnings, simplefilter

from luigi import Task, WrapperTask
from luigi.interface import core
from luigi.task import flatten

from sharp.data.files.base import FileTarget
from sharp.data.files.config import output_root

log = getLogger(__name__)


def clear_all_output():
    if output_root.exists():
        # Ignore the sporadic (and wrong)
        # "OSError: [WinError 145] The directory is not empty: ..."
        rmtree(output_root, ignore_errors=True)
        log.info(f"Cleared directory {output_root}.")


def clear_output(task: Task):
    """
    Recursively expands WrapperTasks until it finds tasks that output one or
    more FileTargets. Then deletes these FileTargets.
    """
    if isinstance(task, WrapperTask):
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


def config_logging():
    # Luigi hasn't initalised logging yet when this is called in __main__.py,
    # so we apply the config file ourselves.
    fileConfig(core().logging_conf_file, disable_existing_loggers=False)


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
