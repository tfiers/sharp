from datetime import date
from logging import Logger, getLogger
from logging.config import dictConfig
from os import environ
from shutil import rmtree
from typing import Iterable, Union

import toml

from sharp.config.spec import config_dir


log = getLogger(__name__)


def validate_config():
    """ Try to load the user specified config. """
    try:
        from sharp.config.load import config
    except ImportError as err:
        raise UserWarning(
            "Possible circular import. Make sure the Tasks you want to run are "
            "imported *inside* the `get_tasks` method of your config.py > "
            "SharpConfig class (and not at the top of the file)."
        ) from err
    return config


def init_log() -> Logger:
    # Luigi hasn't initalised logging yet when this is called in __main__.py,
    # so we apply the logging config ourselves.
    from sharp.config.load import config, output_root

    dictConfig(config.logging)
    log = getLogger("sharp")
    log.info(f"It's {date.today():%b %d, %Y}.")
    log.info(f"Config directory: {config_dir}")
    log.info(f"Output root directory: {output_root}")
    return log


def setup_luigi_config():
    """ Auto-generate a luigi.toml file to configure Luigi. """

    from sharp.config.load import config
    from sharp.config.spec import config_dir

    luigi_config_path = config_dir / "luigi.toml"
    luigi_config = {
        "core": {"scheduler_host": config.luigi_scheduler_host},
        "worker": {
            "keep_alive": True,
            "task_process_context": "",  # Suppress a luigi bug warning.
        },
        "logging": config.logging,
    }
    with open(luigi_config_path, "w") as f:
        toml.dump(luigi_config, f)
    environ["LUIGI_CONFIG_PATH"] = str(luigi_config_path)
    environ["LUIGI_CONFIG_PARSER"] = "toml"


def clear_all_output():
    from sharp.config.load import output_root

    if output_root.exists():
        # Ignore the sporadic (and wrong)
        # "OSError: [WinError 145] The directory is not empty: ..."
        rmtree(output_root, ignore_errors=True)
        log.info(f"Cleared directory {output_root}.")


def clear_output(task):
    """
    Recursively expands tasks without outputs until it finds tasks that output
    one or more FileTargets. Then deletes these FileTargets.
    """
    from luigi import Task
    from luigi.task import flatten
    from sharp.data.files.base import FileTarget

    task: Task

    def clear(target: Union[FileTarget, Iterable[FileTarget]]):
        if isinstance(target, FileTarget):
            target.delete()
        else:
            try:
                for t in target:
                    clear(t)
            except TypeError:
                return

    if task.output():
        clear(task.output())
    else:
        for dependency in flatten(task.requires()):
            clear_output(dependency)
