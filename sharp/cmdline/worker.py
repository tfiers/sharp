"""
Reads the config dir from CLI input for global use in the application.
Loads the user-customized config.py file, sets up logging, and runs tasks.
"""
from socket import gethostname
from logging import Logger, getLogger
from logging.config import dictConfig
from os import chdir, getpid
from pathlib import Path
from shutil import rmtree
from typing import Iterable, Union

from click import argument, command, echo, option

import sharp.config.directory
from sharp.cmdline.util import write_luigi_config
from sharp.config.spec import CONFIG_FILENAME
from sharp.config.types import ConfigError


log = getLogger(__name__)


@command(
    short_help="Start a process that runs tasks.", options_metavar="<options>"
)
@argument("config_directory")
@option(
    "-l",
    "--local-scheduler",
    default=False,
    help="Use an in-process task scheduler. Useful for testing.",
    is_flag=True,
)
@option(
    "--clear-last",
    default=False,
    help=(
        "Remove the output of the final tasks. Only works reliably if these"
        " outputs already exist."
    ),
    is_flag=True,
)
@option(
    "--clear-all",
    default=False,
    help='Empty the entire "output_dir" and "shared_output_dir" directories.',
    is_flag=True,
)
def worker(
    config_directory: str,
    local_scheduler: bool,
    clear_last: bool,
    clear_all: bool,
):
    """
    Reads the "config.py" file in the given CONFIG_DIRECTORY. Runs the tasks
    specified therein, and their dependencies.
    
    Optionally forces tasks to re-run, even if they have been completed
    previously, by deleting their output files before running the tasks.
    """
    #
    # Note: we do not start more than one luigi worker (like e.g.  "luigi
    # --workers 2") in this command, as this yields multiprocessing bugs in luigi
    # / PyTorch / Python. See the README for how to run tasks in parallel.
    #
    config_dir = Path(config_directory).expanduser().resolve().absolute()
    sharp.config.directory.config_dir = config_dir
    config_file = config_dir / CONFIG_FILENAME
    if not config_file.exists():
        echo(f"Could not find file {config_file}.")
        echo(
            f'You can run "sharp config {config_dir}" to generate such a file.'
        )
        return

    # Make sure e.g. log files are generated in the correct directory.
    chdir(str(config_dir))

    config = load_sharp_config()
    log = init_log()

    write_luigi_worker_config()

    log.info("Importing luigi")

    from luigi import build, RPCError

    log.info("Luigi read config file")

    if clear_all:
        log.info("Clearing entire output directories, if they exist.")
        clear_all_output()

    log.info("Importing tasks to run...")
    from sharp.config.util import get_tasks_tuple

    tasks_to_run = get_tasks_tuple(config)
    log.info("Done importing tasks to run.")

    if clear_last:
        for task in tasks_to_run:
            clear_output(task)

    try:
        scheduling_succeeded = build(
            tasks_to_run, local_scheduler=local_scheduler
        )
    except RPCError as err:
        raise ConfigError(
            "Could not connect to centralized Luigi task scheduler. Either run"
            ' the "sharp worker" command with option "--local-scheduler", or see'
            ' the "Parallelization" section of the sharp README on how to start'
            ' a centralized scheduler. Check whether the "scheduler_url" option'
            " option of your config is correctly set."
        ) from err

    log.info(
        "Luigi worker has no more tasks to run."
        " Shutting down Python process."
    )


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
    from sharp.config.load import config, output_root

    dictConfig(config.logging)
    log = getLogger("sharp")
    config_dir = sharp.config.directory.config_dir
    log.info(
        f'Succesfully parsed sharp configuration from {config_dir / "config.py"}'
    )
    log.info(f"Output root directory: {output_root}")
    return log


def write_luigi_worker_config():
    """ Auto-generate a luigi.toml file to configure Luigi workers. """

    # We provide a unique filename for the Luigi config file for each "sharp
    # worker" process. This is to avoid the situation where in one process,
    # luigi is reading the config file, while it another process is halfway in
    # the process of overwriting it.

    from sharp.config.load import config

    output_dir = sharp.config.directory.config_dir / ".luigi-config"
    write_luigi_config(
        output_dir,
        {
            "core": {
                "default-scheduler-url": config.scheduler_url,
                "rpc-retry-attempts": 2 * 60,
                # When the network is down, retry connecting to the scheduler every
                # 30 seconds for this many attempts (instead of the default 3
                # attempts).
            },
            "worker": {
                "keep_alive": True,
                "task_process_context": "",  # Suppress a Luigi bug warning.
            },
            "logging": config.logging,
        },
        filename=f"{gethostname()}__{getpid()}.toml",
    )


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


if __name__ == "__main__":
    # For testing in PyCharm
    worker()
