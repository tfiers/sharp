from pathlib import Path
from shutil import copyfile

from click import argument, command, echo

from sharp.config.default import config as default_config
from sharp.config.spec import CONFIG_FILENAME


@command(
    short_help="Create a new run configuration.", options_metavar="<options>"
)
@argument("directory")
def config(directory: str):
    """
    Creates the given DIRECTORY, and copies a customizable "config.py" file to
    it.
    
    After the default settings in the new "config.py" file have been edited to
    suit your needs, pass the DIRECTORY to "sharp worker". This new directory
    will then be used to read the run configuration from, store log files, and
    (optionally) store task output files.
    """
    config_dir = Path(directory).expanduser().absolute()
    try:
        config_dir.mkdir(parents=True)
    except FileExistsError:
        echo(f"The provided directory ({config_dir}) already exists.")
        return
    echo(f"Created sharp run directory {config_dir}")
    copyfile(default_config.__file__, config_dir / CONFIG_FILENAME)
    echo(f'Created new, customizable "config.py" file in this directory.')
