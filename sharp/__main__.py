"""
Provide a command line interface to run tasks in this package.

Usage:

    $ python -m sharp
    $ python -m sharp --help
"""
from os import environ

from click import command, option


try:
    from sharp.config.load import config
    from sharp.config.spec import config_dir
except ImportError as err:
    raise UserWarning(
        "Possible circular import. Make sure the Tasks you want to run are "
        "imported *inside* the `get_tasks` method of your config.py > "
        "SharpConfig class (and not at the top of the file)."
    ) from err


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
    environ["LUIGI_CONFIG_PARSER"] = "toml"
    environ["LUIGI_CONFIG_PATH"] = str(config_dir / "luigi.toml")
    # Now we can import from luigi (and from sharp.util.startup, which imports
    # luigi itself):

    from luigi import build
    from sharp.util.startup import clear_all_output, clear_output, init_log

    log = init_log()
    log.info("Importing tasks to run...")
    tasks_to_run = config.get_tasks_tuple()
    log.info("Done importing tasks.")
    if clear_all:
        clear_all_output()
    if clear_last:
        for task in tasks_to_run:
            clear_output(task)
    build(tasks_to_run, local_scheduler=local_scheduler)


if __name__ == "__main__":
    run()
