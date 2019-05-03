from datetime import date
from logging import Logger, getLogger
from logging.config import dictConfig
from shutil import rmtree
from typing import Iterable, Union

from luigi import Task
from luigi.task import flatten

from sharp.config.load import config, output_root
from sharp.config.spec import config_dir
from sharp.data.files.base import FileTarget

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


def init_log() -> Logger:
    # Luigi hasn't initalised logging yet when this is called in __main__.py,
    # so we apply the logging config ourselves.
    dictConfig(config.logging)
    log = getLogger("sharp")
    log.info(f"It's {date.today():%b %d, %Y}.")
    log.info(f"Config directory: {config_dir}")
    log.info(f"Output root directory: {output_root}")
    return log
