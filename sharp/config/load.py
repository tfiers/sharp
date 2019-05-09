"""
Locates, loads & initializes a global config object.

Importing from this module (e.g. importing the `config` object, or the
`output_root` path) will attempt to load the user-defined `config.py` file.
"""

from os import environ
from pathlib import Path
from sys import path

from sharp.config.spec import CONFIG_DIR_ENV_VAR, SharpConfigBase, config_dir
from sharp.data.types.config import ConfigError

path.insert(0, str(config_dir))

try:
    # Try importing from a file named `config.py`.
    #
    # Instruction for the PyCharm editor:
    # noinspection PyUnresolvedReferences
    from config import SharpConfig
except ModuleNotFoundError as err:
    if CONFIG_DIR_ENV_VAR in environ:
        msg = (
            f"Could not find file `config.py` in the directory set by the "
            f"{CONFIG_DIR_ENV_VAR} environment variable (which is {config_dir})."
        )
    else:
        msg = (
            f"Could not find file `config.py` in the directory where the "
            f"`python` process is run from ({config_dir}). You can specify an "
            f"explicit directory to look for a `config.py` file by setting the "
            f"{CONFIG_DIR_ENV_VAR} environment variable."
        )
    raise ConfigError(msg) from err
except ImportError as err:
    raise ConfigError(
        "Your custom `config.py` must define a class named `SharpConfig`."
    ) from err

try:
    # A global config object, for use in any Task:
    config: SharpConfigBase = SharpConfig()
except Exception as err:
    raise ConfigError(
        "Could not initialise config.SharpConfig. "
        "See preceding exception for details."
    ) from err


# Some more global config objects
output_root: Path = config.output_dir
shared_output_root: Path = config.shared_output_dir
intermediate_output_dir: Path = output_root / "intermediate"
final_output_dir: Path = output_root / "final"
