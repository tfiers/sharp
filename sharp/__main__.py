"""
Provide a command line interface to run tasks in this package.

Usage:

    $ python -m sharp

"""
from datetime import date
from logging import getLogger

from click import command, option
from luigi import build

from sharp.data.files.config import data_config, output_root
from sharp.tasks.config import get_tasks_to_run
from sharp.util import clear_all_output, clear_output, config_logging


@command()
@option(
    "--clear-last",
    default=False,
    help="Remove the output of the final tasks.",
    is_flag=True,
)
@option(
    "--clear-all",
    default=False,
    help="Empty the entire `output_dir` directory.",
    is_flag=True,
)
@option(
    "--local-scheduler",
    default=False,
    help="Use an in-memory central scheduler. Useful for testing.",
    is_flag=True,
)
def run(clear_last: bool, clear_all: bool, local_scheduler: bool):
    """
    Run the tasks specified in `__main__.py`, by starting a luigi worker
    process. Optionally force tasks to re-run, even if they have been completed
    previously, by deleting their output files before starting luigi.

    Note: we do not start more than one luigi worker (like e.g.  `luigi
    --workers 2`), as this yields multiprocessing bugs in luigi / PyTorch /
    Python. Just start multiple Python processes manually (i.e. calls to this
    module) to run multiple tasks in parallel.
    """
    config_logging()
    log = getLogger("sharp")
    log.info(f"Hello, it's {date.today():%b %d, %Y}.")
    log.info(f"Output root directory: {output_root}")
    log.info(f"Data config: {data_config}")
    tasks = get_tasks_to_run()
    if clear_all:
        clear_all_output()
    if clear_last:
        for task in tasks:
            clear_output(task)
    build(tasks, local_scheduler=local_scheduler)


if __name__ == "__main__":
    run()
