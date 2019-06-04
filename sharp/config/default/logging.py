from datetime import datetime
from logging import Formatter, LogRecord
from logging.handlers import TimedRotatingFileHandler
from os import environ, getenv, getpid
from pathlib import Path
from socket import gethostname

from math import ceil


LOGGER_NAME_LENGTH_UNIT = len("sharp.cmdline.util")

PER_PROCESS_LOGFILES_DIR = Path("logs/per-process")
SINGLE_WORKER_LOGFILES_DIR = Path("logs")

running_as_slurm_task = True if "SLURM_JOB_ID" in environ else False
if running_as_slurm_task:
    job = getenv("SLURM_JOB_ID")
    node = getenv("SLURM_NODEID")
    task = getenv("SLURM_LOCALID")
    slurm_task_ID = f"j{job}.n{node}.t{int(task):02d}"
    filename_prefix = f'{slurm_task_ID.replace(".", "_")}__'
else:
    filename_prefix = ""

# When doing work in subprocess, still use PID of parent process in logs.
host = gethostname()
pid = getpid()

per_process_log_filename = f"{filename_prefix}{host}__PID_{pid}.log"


class SharpFormatter(Formatter):
    def __init__(self, mention_process: bool):
        self.mention_process = mention_process

    def format(self, r: LogRecord):
        parts = [datetime.now().isoformat(sep=" ", timespec="milliseconds")]
        if self.mention_process:
            parts += [f"{host}.PID_{pid: <5}"]
            # Example: "compute01.PID_45904"
            if running_as_slurm_task:
                parts += [slurm_task_ID]
        num_length_units = ceil(len(r.name) / LOGGER_NAME_LENGTH_UNIT)
        parts += [
            f"{r.name: >{num_length_units * LOGGER_NAME_LENGTH_UNIT}}",
            f"{r.levelname}: {r.getMessage()}",
        ]
        return f"{' | '.join(parts)}"


def get_formatter(**kwargs):
    return SharpFormatter(**kwargs)


class DailyRotatingLogFile(TimedRotatingFileHandler):
    def __init__(self, file_path: str):
        log_directory = Path(file_path).parent
        log_directory.mkdir(parents=True, exist_ok=True)
        super().__init__(file_path, when="D")


def get_logging_config(multiple_workers: bool):
    if multiple_workers:
        console_fmt = "multiprocess"
        file_path = str(PER_PROCESS_LOGFILES_DIR / per_process_log_filename)
    else:
        console_fmt = "single_process"
        file_path = str(SINGLE_WORKER_LOGFILES_DIR / "sharp.log")

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "root": {
            # Root logger needs to be specified outside of the "loggers" key.
            "level": "INFO",
            "handlers": ["console", "file"],
        },
        "loggers": {"luigi": {"level": "DEBUG"}, "sharp": {"level": "DEBUG"}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                # Print to stdout instead of the default stderr, so that we do
                # not get an ugly red background in PyCharm or Jupyter.
                "stream": "ext://sys.stdout",
                "formatter": console_fmt,
            },
            "file": {
                "class": "sharp.config.default.logging.DailyRotatingLogFile",
                "file_path": file_path,
                "formatter": "single_process",
            },
        },
        "formatters": {
            "multiprocess": {
                "()": "sharp.config.default.logging.get_formatter",
                "mention_process": True,
            },
            "single_process": {
                "()": "sharp.config.default.logging.get_formatter",
                "mention_process": False,
            },
        },
    }
