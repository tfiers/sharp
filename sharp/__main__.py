"""
Provide a command line interface to run tasks in this package.

Usage:

    $ python -m sharp
    $ python -m sharp --help
"""

# Flag when we have entered our own code.
print("Welcome to the sharp CLI.")

from click import command, option
from sharp.util.startup import (
    clear_all_output,
    clear_output,
    init_log,
    setup_luigi_config,
    validate_config,
)


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
    Run the tasks specified in `config.SharpConfig.get_tasks`, by starting a
    luigi worker process. Optionally force tasks to re-run, even if they have
    been completed previously, by deleting their output files before starting
    luigi.

    Note: we do not start more than one luigi worker (like e.g.  `luigi
    --workers 2`) in this command, as this yields multiprocessing bugs in luigi
    / PyTorch / Python. See the ReadMe for how to run tasks in parallel.
    """
    config = validate_config()
    log = init_log()
    log.info("Generating luigi config..")
    setup_luigi_config()
    log.info("Done")
    # Now we can import from luigi.

    if clear_all:
        log.info("Clearing entire output directory, if it exists.")
        clear_all_output()
    log.info("Importing tasks to run...")
    tasks_to_run = config.get_tasks_tuple()
    log.info("Done importing tasks to run.")
    if clear_last:
        for task in tasks_to_run:
            clear_output(task)
    from luigi import build

    build(tasks_to_run, local_scheduler=local_scheduler)


if __name__ == "__main__":
    run()
