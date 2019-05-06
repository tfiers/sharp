LOGGING_CONFIG = dict(
    version=1,
    disable_existing_loggers=False,
    formatters={
        "standard": {
            "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        },
        # When using the "time_only" formatter, make sure to log the full date
        # at the start of the log.
        "time_only": {
            "format": "%(asctime)s | %(message)s",
            "datefmt": "%H:%M:%S",
        },
    },
    loggers={
        "root": {
            # [Linebreak for auto-code formatter Black]
            "qualname": "root",
            "level": "INFO",
            "handlers": ["console"],
        },
        "luigi_scheduler": {
            "qualname": "luigi",
            "level": "INFO",
            "propagate": True,
            "handlers": ["file_luigi"],
        },
        "luigi_worker": {
            "qualname": "luigi-interface",
            "level": "INFO",
            "propagate": True,
            "handlers": ["file_luigi"],
        },
        # For logging in our own package.
        "sharp": {
            "qualname": "sharp",
            "level": "INFO",
            "propagate": True,
            "handlers": ["file_sharp"],
        },
    },
    handlers={
        "console": {
            "class": "logging.StreamHandler",
            # Print to stdout instead of stderr, so we do not get an ugly red
            # background for luigi output in PyCharm or Jupyter.
            "stream": "sys.stdout",
            # Remove next line to log the message only.
            "formatter": "time_only",
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
            "formatter": "time_only",
        },
    },
)
