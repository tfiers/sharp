from functools import partial
from logging import getLogger
from os import environ
from pathlib import Path

import click
import toml

from sharp.config.types import ConfigDict


log = getLogger(__name__)


def resolve_path_str(path: str) -> Path:
    return Path(path).expanduser().resolve().absolute()


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


# Change default Click behaviour: wider help texts, friendlier looking Usage
# strings, and display of default argument values.
command_kwargs = dict(
    options_metavar="<options>", context_settings=dict(max_content_width=100)
)
sharp_command = partial(click.command, **command_kwargs)
sharp_command_group = partial(
    click.group, **command_kwargs, subcommand_metavar="<command>"
)
option = partial(click.option, show_default=True)
