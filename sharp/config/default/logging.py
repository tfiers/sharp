from datetime import datetime
from logging import Formatter, LogRecord
from os import environ, getenv, getpid
from pathlib import Path
from socket import gethostname

from math import ceil


LOGGER_NAME_LENGTH_UNIT = len("sharp.cmdline.util")

logfile_dirs = {
    "per_process": Path("logs/per-process"),
    "multiprocess": Path("logs/multiprocess"),
    "single_worker": Path("logs"),
}


process_ID = f"{gethostname()}.{getpid()}"
# Example: "compute01.45904"

running_as_slurm_task = True if "SLURM_JOB_ID" in environ else False
if running_as_slurm_task:
    job = getenv("SLURM_JOB_ID")
    node = getenv("SLURM_NODEID")
    task = getenv("SLURM_LOCALID")
    slurm_task_ID = f"j{job}.n{node}.t{int(task):02d}"
    filename_stem = f"{process_ID}__{slurm_task_ID}"
else:
    filename_stem = process_ID

per_process_log_filename = f'{filename_stem.replace(".", "_")}.log'


class ClusterFormatter(Formatter):
    def __init__(self, mention_process: bool):
        self.mention_process = mention_process

    def format(self, r: LogRecord):
        parts = [datetime.now().isoformat(sep=" ", timespec="milliseconds")]
        if self.mention_process:
            parts += [f"{process_ID: <15}"]
            if running_as_slurm_task:
                parts += [slurm_task_ID]
        num_length_units = ceil(len(r.name) / LOGGER_NAME_LENGTH_UNIT)
        parts += [
            f"{r.name: >{num_length_units * LOGGER_NAME_LENGTH_UNIT}}",
            f"{r.levelname}: {r.getMessage()}",
        ]
        return f"{' | '.join(parts)}"


def get_cluster_formatter(**kwargs):
    return ClusterFormatter(**kwargs)


def mkdir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def get_logging_config(multiple_workers: bool):
    kwargs_for_all_filehandlers = {
        "class": "logging.handlers.TimedRotatingFileHandler",
        "when": "D",
    }
    if multiple_workers:
        console_fmt = "multiprocess"
        file_handlers = {
            "multiprocess_file": {
                "filename": str(logfile_dirs["multiprocess"] / "sharp.log"),
                "formatter": "multiprocess",
                **kwargs_for_all_filehandlers,
            },
            "per_process_file": {
                "filename": str(
                    logfile_dirs["per_process"] / per_process_log_filename
                ),
                "formatter": "single_process",
                **kwargs_for_all_filehandlers,
            },
        }
        mkdir(logfile_dirs["multiprocess"])
        mkdir(logfile_dirs["per_process"])
    else:
        console_fmt = "single_process"
        file_handlers = {
            "single_worker_file": {
                "filename": str(logfile_dirs["single_worker"] / "sharp.log"),
                "formatter": "single_process",
                **kwargs_for_all_filehandlers,
            }
        }
        mkdir(logfile_dirs["single_worker"])
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "root": {
            # Root logger needs to be specified outside of a 'loggers' key.
            "level": "INFO",
            "handlers": ["console", *file_handlers.keys()],
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                # Print to stdout instead of the default stderr, so that we do
                # not get an ugly red background in PyCharm or Jupyter.
                "stream": "ext://sys.stdout",
                "formatter": console_fmt,
            },
            **file_handlers,
        },
        "formatters": {
            "multiprocess": {
                "()": "sharp.config.default.logging.get_cluster_formatter",
                "mention_process": True,
            },
            "single_process": {
                "()": "sharp.config.default.logging.get_cluster_formatter",
                "mention_process": False,
            },
        },
    }
