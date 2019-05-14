msg_format = "%(asctime)s | %(name)s | %(levelname)s: %(message)s"

LOGGING_CONFIG = dict(
    version=1,
    disable_existing_loggers=False,
    formatters={
        "standard": {"format": msg_format},
        "time_only": {"format": msg_format, "datefmt": "%H:%M:%S"},
        # When using the "time_only" formatter, make sure to log the full date
        # at the start of the log.
    },
    root={
        # Root logger neds to be specified separately.
        "level": "INFO",
        "handlers": ["console"],
    },
    loggers={
        # Luigi scheduler
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
            "formatter": "standard",
        },
        "file_luigi": {
            "class": "logging.FileHandler",
            "filename": "luigi.log",
            "mode": "a",
            "formatter": "standard",
        },
        # Mode 'w' clears existing file. Mode 'a' appends.
        "file_sharp": {
            "class": "logging.FileHandler",
            "filename": "sharp.log",
            "mode": "a",
            "formatter": "standard",
        },
    },
)
