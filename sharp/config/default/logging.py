from datetime import datetime
from logging import Formatter, LogRecord
from os import getenv
from socket import gethostname

from sharp.util.misc import format_duration

hostname = gethostname()
node_ID = getenv("SLURM_NODEID")
task_ID = getenv("SLURM_LOCALID")

LONG_MODULE_NAME = "sharp.data.hardcoded.filters.literature"
LONG_LOG_LEVEL = "CRITICAL"


class ClusterFormatter(Formatter):
    def format(self, r: LogRecord):
        rel_time = format_duration(r.relativeCreated / 1000, ms_digits=3)
        metadata = (
            f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            rel_time,
            f"{r.name: >{len(LONG_MODULE_NAME)}}",
        )
        if node_ID is not None:
            metadata += (f"worker {node_ID}.{int(task_ID):02d}",)
        return f"{' | '.join(metadata)} | {r.levelname+':': <{len(LONG_LOG_LEVEL)}} {r.getMessage()}"


get_formatter = lambda: ClusterFormatter()


LOGGING_CONFIG = dict(
    version=1,
    disable_existing_loggers=False,
    formatters={
        "cluster": {"()": "sharp.config.default.logging.get_formatter"}
    },
    root={
        # Root logger neds to be specified separately.
        "level": "INFO",
        "handlers": ["console"],
    },
    loggers={
        # Luigi main and scheduler.
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
