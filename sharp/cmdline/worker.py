"""
Reads the config dir from CLI input for global use in the application.
Loads the user-customized config.py file, sets up logging, and runs tasks.
"""
import os
from logging import Logger, getLogger
from logging.config import dictConfig
from multiprocessing import cpu_count
from os import chdir, getpid
from shutil import rmtree
from socket import gethostname
from time import time
from typing import Iterable, Union
from urllib.error import URLError
from urllib.request import urlopen

from click import argument, echo

import sharp.config.directory
from sharp.cmdline.util import (
    option,
    resolve_path_str,
    sharp_command,
    write_luigi_config,
)
from sharp.config.spec import CONFIG_FILENAME, SharpConfig
from sharp.config.types import ConfigError
from sharp.util.misc import format_duration, linearize


log = getLogger(__name__)

if os.name == "nt":
    # On Windows, don't use multiprocessing (cannot fork process on Windows).
    use_subprocesses = False
else:
    # On Unix, work in one or more subprocesses. Even when only one subprocess
    # is used, this has the advantage that memory is not only accumulating
    # (Python doesn't normally release memory back to OS).
    use_subprocesses = True


@sharp_command(short_help="Start a process that runs tasks.")
@argument("config_directory")
@option(
    "-l",
    "--local-scheduler",
    default=False,
    help="Use an in-process task scheduler. Useful for testing.",
    is_flag=True,
)
@option(
    "-n",
    "--num-subprocesses",
    type=int,
    default=cpu_count(),
    help=(
        "Number of subprocess that are launched, to run tasks in parallel."
        f" Default is number of CPU's. Ignored on Windows (see README)."
    ),
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
    num_subprocesses: int,
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
    config_dir = resolve_path_str(config_directory)
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
    if config.scheduler_url is None:
        local_scheduler = True
    if not local_scheduler:
        check_scheduling_server(config)
    write_luigi_worker_config()
    log.info("Importing luigi")

    from luigi import build

    log.info("Luigi read config file")
    if clear_all:
        log.info("Clearing entire output directories, if they exist.")
        clear_all_output()
    log.info("Importing tasks to run...")
    t0 = time()
    from sharp.config.util import get_tasks_tuple

    tasks_to_run = get_tasks_tuple(config)
    log.info(
        f"Done importing tasks to run. Took {format_duration(time() - t0)}"
    )
    if clear_last:
        for task in tasks_to_run:
            log.info(f"Removing output of task {task}")
            clear_output(task)

    build(
        tasks_to_run,
        local_scheduler=local_scheduler,
        workers=num_subprocesses if use_subprocesses else 1,
    )

    log.info(
        "Luigi worker has no more tasks to run. Shutting down Python process."
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


def check_scheduling_server(config: SharpConfig):
    try:
        urlopen(config.scheduler_url, timeout=2)
    except URLError as err:
        msg = linearize(
            f"""Could not connect to centralized Luigi task scheduler at
            "{config.scheduler_url}". Either run the "sharp worker" command
            with option "--local-scheduler", or see the "Visualization &
            Cluster computing" section of the sharp README on how to start a
            centralized scheduler. Check whether the "scheduler_url" option
            of your config is correctly set."""
        )
        raise ConfigError(msg) from err


def write_luigi_worker_config():
    """ Auto-generate a luigi.toml file to configure Luigi workers. """

    # We provide a unique filename for the Luigi config file for each "sharp
    # worker" process. This is to avoid the situation where in one process,
    # luigi is reading the config file, while it another process is halfway in
    # the process of overwriting it.

    from sharp.config.load import config

    output_dir = sharp.config.directory.config_dir / ".temp-luigi-config-files"
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
                "force_multiprocessing": True if use_subprocesses else False,
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


# For testing in PyCharm
if __name__ == "__main__":
    worker()
