"""
Reads the config dir from CLI input for global use in the application.
Loads the user-customized config.py file, sets up logging, and runs tasks.
"""

from pathlib import Path

from click import argument, command, option, echo

from sharp.cli.util import (
    clear_all_output,
    clear_output,
    init_log,
    load_sharp_config,
    setup_luigi_worker_config,
)
from sharp.config.spec import CONFIG_FILENAME
from sharp.config.types import ConfigError


config_dir: Path
# Globally usable object.


@command(short_help="Start a process that runs tasks.")
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
    global config_dir

    config_dir = Path(config_directory).expanduser().resolve().absolute()
    config_file = config_dir / CONFIG_FILENAME
    if not config_file.exists():
        echo(f"Could not find file {config_file}.")
        echo(
            f'You can run "sharp config {config_dir}" to generate such a file.'
        )
        return

    config = load_sharp_config()
    log = init_log()
    setup_luigi_worker_config()
    log.info("Generated Luigi config.")
    # Now we can import from Luigi, which will apply the generated config.

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

    from luigi import build, RPCError

    try:
        scheduling_succeeded = build(
            tasks_to_run, local_scheduler=local_scheduler
        )
    except RPCError as err:
        raise ConfigError(
            "Could not connect to centralized Luigi task scheduler. Either run"
            ' the "sharp worker" command with option "--local-scheduler", or see'
            ' the "Parallelization" section of the sharp README on how to start '
            ' a centralized scheduler. Check whether the "luigi_scheduler_host"'
            " option of your config is correctly set."
        ) from err

    log.info("Luigi worker has no more tasks to complete.")
    log.info("Shutting down Python process.")
