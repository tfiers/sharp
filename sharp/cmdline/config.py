from pathlib import Path
from shutil import copyfile

from click import argument, command, echo, option, confirm

from sharp.config.default import config as default_config
from sharp.config.spec import CONFIG_FILENAME


@command(
    short_help="Create a new run configuration.", options_metavar="<options>"
)
@option(
    "-y",
    "--yes",
    "overwrite",
    help='Overwrite existing "config.py" file without prompting.',
    default=False,
    is_flag=True,
)
@argument("directory")
def config(directory: str, overwrite: bool):
    """
    Creates the given DIRECTORY, and copies a customizable "config.py" file to
    it.
    
    After the default settings in the new "config.py" file have been edited to
    suit your needs, pass the DIRECTORY to "sharp worker". This new directory
    will then be used to read the run configuration from, store log files, and
    (optionally) store task output files.
    """
    config_dir = Path(directory).expanduser().absolute()
    if not config_dir.exists():
        config_dir.mkdir(parents=True)
        echo(f"Created sharp run directory {config_dir}")
    else:
        echo(f"Found existing directory {config_dir}")
    config_file = config_dir / CONFIG_FILENAME
    if not config_file.exists():
        copyfile(default_config.__file__, config_file)
        echo(f'Created new, customizable "config.py" file in this directory.')
    else:
        echo('A "config.py" file already exists in this directory.')
        if overwrite or confirm(
            'Do you want to overwrite it with a new, default "config.py" file?\n'
            '(Existing file will be copied to "old__config.py" first)'
        ):
            copyfile(config_file, config_file.parent / "old__config.py")
            copyfile(default_config.__file__, config_file)
            echo(f'Overwritten existing "config.py" with the default file.')
