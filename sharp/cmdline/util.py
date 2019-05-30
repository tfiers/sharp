from logging import getLogger
from os import environ
from pathlib import Path

import toml

from sharp.config.types import ConfigDict

log = getLogger(__name__)


def write_luigi_config(
    directory: Path, config_dict: ConfigDict, filename: str = "luigi.toml"
):
    directory.mkdir(exist_ok=True, parents=True)
    luigi_config_path = directory / filename
    with open(luigi_config_path, "w") as f:
        toml.dump(config_dict, f)
    environ["LUIGI_CONFIG_PATH"] = str(luigi_config_path)
    environ["LUIGI_CONFIG_PARSER"] = "toml"
    log.info(
        f"Wrote Luigi config file {luigi_config_path} and set Luigi env vars."
    )
