"""
Importing from this module (e.g. the `config` object, or the
`output_root` path) will attempt to load the user-defined `config.py` file.
"""

from os import environ
from pathlib import Path
from sys import path
from warnings import warn

from sharp.config.spec import SharpConfigBase, ConfigError


# Locate, load & initialize config object
# ---------------------------------------

ENV_VAR = "SHARP_CONFIG_DIR"

config_dir = Path(environ.get(ENV_VAR, ".")).absolute()
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


# Validate config
# ---------------

settings = dir(SharpConfigBase)
for name in dir(config):
    if name not in settings:
        warn(f"SharpConfig attribute `{name}` is not a config setting.")
    if name != "tasks_to_run":
        # Should not get tasks_to_run value: this imports tasks too soon.
        value = getattr(config, name)
        if value == NotImplemented:
            raise ConfigError(
                f"The mandatory config setting `{name}` is not set."
            )


# Normalize config
# ----------------

if config.config_id is None:
    config.config_id = str(config_dir)


def _as_absolute_Path(path: str) -> Path:
    ppath = Path(path)
    if ppath.is_absolute():
        return ppath
    else:
        return config_dir / ppath


config.output_dir = _as_absolute_Path(config.output_dir)
config.raw_data_dir = _as_absolute_Path(config.raw_data_dir)


# Some global config values
# -------------------------

output_root: Path = config.output_dir
intermediate_output_dir: Path = output_root / "intermediate"
final_output_dir: Path = output_root / "final"
