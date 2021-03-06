"""
Locates, loads & initializes a global config object.

Importing from this module (e.g. importing the `config` object, or the
`output_root` path) will attempt to load the user-defined `config.py` file.
"""

from pathlib import Path
from sys import path

import sharp.config.directory
from sharp.config.spec import SharpConfig
from sharp.config.types import ConfigError
from sharp.config.util import normalize, validate


config_dir = sharp.config.directory.config_dir

path.insert(0, str(config_dir))

try:
    # Try importing an object named `config` from a file named `config.py`.

    # Instruction for PyCharm code editor:
    # noinspection PyUnresolvedReferences
    from config import config

except TypeError as err:
    raise ConfigError(
        "Your custom `config` object could not be initialized."
        " One of the specified options / arguments may have a wrong name."
        " See preceding exception for details."
    ) from err

if not isinstance(config, SharpConfig):
    raise ConfigError(
        "Your custom `config` object must be an instance of the `SharpConfig`"
        " class from `sharp.config.spec`."
        " See the 'Usage' section from the main README for details."
    )

try:
    validate(config)
except Exception as err:
    raise ConfigError(
        "Your custom `config` object is not to specification."
        " See preceding exception for details."
    ) from err


# Define some globally usable config objects, for use in any task:

config: SharpConfig = normalize(config)
output_root: Path = config.output_dir
shared_output_root: Path = config.shared_output_dir
intermediate_output_dir: Path = output_root / "intermediate"
final_output_dir: Path = output_root / "final"
