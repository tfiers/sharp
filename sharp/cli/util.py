from os import environ
from pathlib import Path

import toml

from sharp.config.types import ConfigDict


def setup_luigi_config(directory: Path, config_dict: ConfigDict):
    luigi_config_path = directory / "luigi.toml"
    with open(luigi_config_path, "w") as f:
        toml.dump(config_dict, f)
    environ["LUIGI_CONFIG_PATH"] = str(luigi_config_path)
    environ["LUIGI_CONFIG_PARSER"] = "toml"
