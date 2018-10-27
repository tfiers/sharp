"""
Locates, loads & initializes a global config object.

Importing from this module (e.g. importing the `config` object, or the
`output_root` path) will attempt to load the user-defined `config.py` file.
"""

from os import environ
from pathlib import Path
from sys import path

from sharp.config.spec import ConfigError, ENV_VAR, SharpConfigBase, config_dir


path.insert(0, str(config_dir))

try:
    # noinspection PyUnresolvedReferences
    from config import SharpConfig
except ModuleNotFoundError as err:
    if ENV_VAR in environ:
        msg = (
            f"Could not find file `config.py` in the directory set by the "
            f"{ENV_VAR} environment variable ({config_dir})."
        )
    else:
        msg = (
            f"Could not find file `config.py` in the directory where the "
            f"`python` process is run from ({config_dir}). You can specify an "
            f"explicit directory to look for this file by setting the "
            f"{ENV_VAR}  environment variable."
        )
    raise ConfigError(msg) from err

try:
    # A global config object, for use in any Task:
    config: SharpConfigBase = SharpConfig()
except Exception as err:
    raise ConfigError("Could not initialise config.SharpConfig.") from err


output_root: Path = config.output_dir
intermediate_output_dir: Path = output_root / "intermediate"
final_output_dir: Path = output_root / "final"
