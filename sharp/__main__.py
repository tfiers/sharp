"""
Provide a command line interface to run tasks in this package.

Usage:

    $ python -m sharp
    $ python -m sharp --help
"""

# Flag when we have entered our own code.
import toml

print("Welcome to the sharp CLI.")


from os import environ

from click import command, option


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
    # Try to load the user specified config.
    try:
        from sharp.config.load import config
        from sharp.config.spec import config_dir
    except ImportError as err:
        raise UserWarning(
            "Possible circular import. Make sure the Tasks you want to run are "
            "imported *inside* the `get_tasks` method of your config.py > "
            "SharpConfig class (and not at the top of the file)."
        ) from err

    # Auto-generate a luigi.toml file to configure Luigi.
    luigi_config_path = config_dir / "luigi.toml"
    luigi_config = dict(
        core=dict(scheduler_host=config.luigi_scheduler_host),
        worker=dict(
            keep_alive=True,
            task_process_context="",  # Suppress a luigi bug warning.
        ),
        logging=config.logging,
    )
    with open(luigi_config_path, "w") as f:
        toml.dump(luigi_config, f)
    environ["LUIGI_CONFIG_PATH"] = str(luigi_config_path)
    environ["LUIGI_CONFIG_PARSER"] = "toml"

    # Now we can import from luigi (and from sharp.util.startup, which imports
    # luigi):

    from luigi import build
    from sharp.util.startup import clear_all_output, clear_output, init_log

    log = init_log()
    if clear_all:
        clear_all_output()
    log.info("Importing tasks to run...")
    tasks_to_run = config.get_tasks_tuple()
    log.info("Done importing tasks to run.")
    if clear_last:
        for task in tasks_to_run:
            clear_output(task)
    build(tasks_to_run, local_scheduler=local_scheduler)


if __name__ == "__main__":
    run()
