from datetime import datetime
from logging import Formatter, LogRecord
from os import getenv, getpid
from socket import gethostname

from math import ceil


node_ID = getenv("SLURM_NODEID")
task_ID = getenv("SLURM_LOCALID")

LOGGER_NAME_LENGTH_UNIT = len("sharp.cmdline.util")


class ClusterFormatter(Formatter):
    def format(self, r: LogRecord):
        parts = [datetime.now().isoformat(sep=" ", timespec="milliseconds")]
        if node_ID is not None:
            process = f"{gethostname()}.{getpid()}"
            # Example: "compute01.45904"
            parts += [f"{process: <15}", f"n{node_ID}.t{int(task_ID):02d}"]
        k = ceil(len(r.name) / LOGGER_NAME_LENGTH_UNIT)
        parts += [
            f"{r.name: >{k * LOGGER_NAME_LENGTH_UNIT}}",
            f"{r.levelname}: {r.getMessage()}",
        ]
        return f"{' | '.join(parts)}"


get_cluster_formatter = lambda: ClusterFormatter()


LOGGING_CONFIG = dict(
    version=1,
    disable_existing_loggers=False,
    formatters={
        "cluster": {"()": "sharp.config.default.logging.get_cluster_formatter"}
    },
    root={
        # Root logger needs to be specified separately.
        "level": "INFO",
        "handlers": ["console"],
    },
    loggers={
        # Luigi main, scheduler, server.
        "luigi": {
            "level": "INFO",
            "propagate": True,
            "handlers": ["file_luigi"],
        },
        # Luigi worker
        "luigi-interface": {
            "level": "INFO",
            "propagate": True,
            "handlers": ["file_luigi"],
        },
        # For logging in our own package.
        "sharp": {
            "level": "DEBUG",
            "propagate": False,
            "handlers": ["file_sharp", "console"],
        },
    },
    handlers={
        "console": {
            "class": "logging.StreamHandler",
            # Print to stdout instead of the default stderr, so that we do not
            # get an ugly red background for luigi output in PyCharm or
            # Jupyter.
            "stream": "ext://sys.stdout",
            "formatter": "cluster",
        },
        "file_luigi": {
            "class": "logging.FileHandler",
            "filename": "luigi.log",
            "mode": "a",
            "formatter": "cluster",
        },
        # Mode 'w' clears existing file. Mode 'a' appends.
        "file_sharp": {
            "class": "logging.FileHandler",
            "filename": "sharp.log",
            "mode": "a",
            "formatter": "cluster",
        },
    },
)
