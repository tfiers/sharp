from logging import Logger, getLogger
from logging.config import dictConfig
from os import environ
from pathlib import Path
from shutil import rmtree
from typing import Iterable, Union

import toml

from sharp.config.types import ConfigDict

log = getLogger(__name__)


def load_sharp_config():
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
    from sharp.cli.worker import config_dir
    from sharp.config.load import config, output_root

    dictConfig(config.logging)
    log = getLogger("sharp")
    log.info(
        f'Succesfully parsed sharp configuration from {config_dir / "config.py"}'
    )
    log.info(f"Output root directory: {output_root}")
    return log


def setup_luigi_worker_config():
    """ Auto-generate a luigi.toml file to configure Luigi workers. """

    from sharp.cli.worker import config_dir
    from sharp.config.load import config

    luigi_config = {
        "core": {
            "scheduler_host": config.luigi_scheduler_host,
            "rpc-retry-attempts": 1,
            # "rpc-retry-attempts": 2 * 60,  todo: get back
            # When the network is down, retry connecting to the scheduler every
            # 30 seconds for this many attempts (instead of the default 3
            # attempts).
        },
        "worker": {
            "keep_alive": True,
            "task_process_context": "",  # Suppress a Luigi bug warning.
        },
        "logging": config.logging,
    }
    setup_luigi_config(config_dir, luigi_config)


def setup_luigi_config(directory: Path, config_dict: ConfigDict):
    luigi_config_path = directory / "luigi.toml"
    with open(luigi_config_path, "w") as f:
        toml.dump(config_dict, f)
    environ["LUIGI_CONFIG_PATH"] = str(luigi_config_path)
    environ["LUIGI_CONFIG_PARSER"] = "toml"


def clear_all_output():
    from sharp.config.load import output_root, shared_output_root

    for dir in (output_root, shared_output_root):
        if dir.exists():
            # Ignore the sporadic (and wrong)
            # "OSError: [WinError 145] The directory is not empty: ..."
            rmtree(dir, ignore_errors=True)
            log.info(f"Cleared directory {dir}.")


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
