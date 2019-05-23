from datetime import datetime
from logging import Formatter, LogRecord
from os import getenv, getpid

from math import ceil

from sharp.util.misc import format_duration


node_ID = getenv("SLURM_NODEID")
task_ID = getenv("SLURM_LOCALID")

LOGGER_NAME_LENGTH_UNIT = len("sharp.cli.scheduler")
LONG_LOG_LEVEL = "CRITICAL"


class ClusterFormatter(Formatter):
    def format(self, r: LogRecord):
        rel_time = format_duration(r.relativeCreated / 1000, ms_digits=3)
        metadata = [f"{datetime.now():%Y-%m-%d %H:%M:%S}", rel_time]
        if node_ID is not None:
            metadata += [
                f"PID {getpid(): >5}",
                f"worker {node_ID}.{int(task_ID):02d}",
            ]
        k = ceil(len(r.name) / LOGGER_NAME_LENGTH_UNIT)
        metadata += [f"{r.name: >{k * LOGGER_NAME_LENGTH_UNIT}}"]
        return f"{' | '.join(metadata)} | {r.levelname+':': <{len(LONG_LOG_LEVEL)}} {r.getMessage()}"


get_cluster_formatter = lambda: ClusterFormatter()


LOGGING_CONFIG = dict(
    version=1,
    disable_existing_loggers=False,
    formatters={
        "cluster": {"()": "sharp.config.default.logging.get_cluster_formatter"}
    },
    root={
        # Root logger neds to be specified separately.
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
            "level": "INFO",
            "propagate": True,
            "handlers": ["file_sharp"],
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
