"""
A CLI to start, stop, and check the Luigi centralized task scheduler
(https://luigi.readthedocs.io/en/stable/central_scheduler.html).
"""
import re
from socket import gethostname
from pathlib import Path
from subprocess import run, PIPE
from time import sleep
from typing import Tuple, Optional

from click import group, option, argument, echo

from sharp.cli.util import setup_luigi_config

PID_FILENAME = "scheduler.pid"
STATE_FILENAME = "scheduler-state.pickle"
LOGDIR_NAME = "logs"


@group(
    short_help="Manage the centralized task scheduler.",
    options_metavar="<options>",
    subcommand_metavar="<command>",
    epilog='Type "sharp scheduler <command> -h" for more help.',
)
def scheduler():
    """
    Manage the state of the centralized Luigi task scheduler.
    See the "Parallelization" section of the sharp README.
    """


@scheduler.command(
    short_help="Setup and start the scheduling server.",
    options_metavar="<options>",
)
@argument("directory")
@option(
    "-p",
    "--port",
    type=int,
    default=8082,
    help="TCP port at which to run the server. Default is 8082 (same as Luigi default).",
)
def start(directory: str, port: int):
    """
    Starts a server daemon (the Luigi centralized task scheduler), using the
    provided DIRECTORY to store server logs, the task history database, a PID
    file, and a pickled scheduler state file.
    """
    scheduler_dir = resolve_dir_arg(directory)
    scheduler_dir.mkdir(exist_ok=True, parents=True)
    pid_file = scheduler_dir / PID_FILENAME
    if pid_file.exists():
        rel_scheduler_dir = scheduler_dir.relative_to(Path.cwd())
        echo(
            f"An existing Luigi scheduler server is already running from {scheduler_dir}.\n"
            f'Check its state and address using "sharp scheduler state {rel_scheduler_dir}"\n'
            f'or stop the server using "sharp scheduler stop {rel_scheduler_dir}"'
        )
    else:
        state_file = scheduler_dir / STATE_FILENAME
        logdir = scheduler_dir / LOGDIR_NAME
        setup_luigi_scheduler_config(scheduler_dir)
        run(
            [
                "luigid",
                "--background",
                "--pidfile",
                str(pid_file),
                "--logdir",
                str(logdir),
                "--state-path",
                str(state_file),
                "--port",
                str(port),
            ]
        )
        echo("Waiting 5 seconds for server startup..\n")
        sleep(5)
        state(directory)


@scheduler.command(options_metavar="<options>")
@argument("directory")
def stop(directory: str):
    """
    Stop the scheduling server.
    """
    scheduler_dir = resolve_dir_arg(directory)
    pid_file = scheduler_dir / PID_FILENAME
    pid, port = get_pid_and_port(scheduler_dir)
    if port is not None:
        run(["kill", "-SIGTERM", pid])
        echo(
            f"Shut down Luigi scheduling server with PID {pid} listening on TCP port {port}."
        )
        pid_file.unlink()
        echo(f"Removed PID file {pid_file}")


@scheduler.command(options_metavar="<options>")
@argument("directory")
def state(directory: str):
    """
    Check status of the scheduling server.
    """
    scheduler_dir = resolve_dir_arg(directory)
    pid, port = get_pid_and_port(scheduler_dir)
    if port is not None:
        echo(
            f"Luigi centralized task scheduler running as server daemon with"
            f" PID {pid}, on TCP port {port}."
        )
        echo(
            f"Check out the task scheduler GUI at http://{gethostname()}:{port}"
        )


def resolve_dir_arg(directory: str):
    return Path(directory).expanduser().resolve().absolute()


def get_pid_and_port(
    scheduler_dir: Path
) -> Tuple[Optional[str], Optional[str]]:
    pid_file = scheduler_dir / PID_FILENAME
    if not pid_file.exists():
        breakpoint()
        echo(f"No Luigi scheduling server running from {scheduler_dir}")
        return (None, None)
    else:
        pid = get_pid(scheduler_dir)
        port = get_port(scheduler_dir)
        if not port:
            echo(
                f"A PID file of the Luigi server was found at {pid_file},\n"
                f" but the process with this PID ({pid}) is not listening on any TCP"
                f" ports. Did you manually stop the server?\nIf not, check the"
                f" error file in {scheduler_dir / LOGDIR_NAME}."
            )
            return (pid, None)
        else:
            return (pid, port)


def get_pid(scheduler_dir: Path):
    with open(scheduler_dir / PID_FILENAME) as f:
        pid = f.read()
    return pid


def get_port(scheduler_dir: Path):
    pid = get_pid(scheduler_dir)
    result = run(["netstat", "-lp"], stdout=PIPE, stderr=PIPE)
    lines = result.stdout.decode().splitlines()
    try:
        line = next(line for line in lines if pid in line)
        port = re.findall(":(\d+)", line)[0]
        return port
    except StopIteration:
        return None


def setup_luigi_scheduler_config(scheduler_directory: Path):
    """
    Auto-generate a luigi.toml file to configure the Luigi scheduling server.
    
    :param scheduler_directory:  An absolute and resolved path.
    """
    luigi_config = {
        "scheduler": {
            "record_task_history": True,
            "state_path": str(scheduler_directory / STATE_FILENAME),
        },
        "task_history": {
            # SqlAlchemy connection string
            "db_connection": f"sqlite:///{scheduler_directory/'task-history.db'}"
        },
    }
    setup_luigi_config(scheduler_directory, luigi_config)
