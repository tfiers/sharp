"""
Provide a command line interface to run tasks in this package.

Usage:

    $ python -m sharp

"""

from click import command, option
from luigi import build

from sharp.tasks.main import TASKS_TO_RUN
from sharp.util import clear_all_output, clear_output, init_log


@command()
@option(
    "--clear-last",
    default=False,
    help="Remove the output of the final tasks. Only works reliably if they exist.",
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
    Run the tasks specified in `tasks/main.py`, by starting a luigi worker
    process. Optionally force tasks to re-run, even if they have been completed
    previously, by deleting their output files before starting luigi.

    Note: we do not start more than one luigi worker (like e.g.  `luigi
    --workers 2`) in this command, as this yields multiprocessing bugs in luigi
    / PyTorch / Python. See the ReadMe for how to run tasks in parallel.
    """
    init_log()
    if clear_all:
        clear_all_output()
    if clear_last:
        for task in TASKS_TO_RUN:
            clear_output(task)
    build(TASKS_TO_RUN, local_scheduler=local_scheduler)


if __name__ == "__main__":
    run()
