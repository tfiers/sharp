from logging import getLogger
from os import environ
from pathlib import Path

import toml

from sharp.config.types import ConfigDict

log = getLogger(__name__)


def setup_luigi_config(directory: Path, config_dict: ConfigDict):
    luigi_config_path = directory / "luigi.toml"
    # When multiprocessing on the cluster, avoid reading partly written config
    # files.
    if luigi_config_path.exists():
        log.info(
            f"Luigi config file already exists at {luigi_config_path}."
            f" Not generating a new file."
        )
    else:
        with open(luigi_config_path, "w") as f:
            toml.dump(config_dict, f)
    environ["LUIGI_CONFIG_PATH"] = str(luigi_config_path)
    environ["LUIGI_CONFIG_PARSER"] = "toml"
