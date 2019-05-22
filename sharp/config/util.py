from pathlib import Path
from typing import Tuple

from typeguard import check_type

import sharp.config.directory
from sharp.config.spec import SharpConfig
from sharp.config.types import ConfigError, LuigiTask


def validate(config: SharpConfig):
    for name in dir(config):
        value = getattr(config, name)
        expected_type = SharpConfig.__annotations__.get(name)
        if expected_type:
            try:
                check_type(name, value, expected_type)
            except TypeError as e:
                raise ConfigError(
                    f'The config setting "{name}" has an incorrect type. '
                    f'Expected type: "{expected_type}". '
                    f'Got type: "{type(value)}".'
                    f"See preceding exception for details."
                ) from e


def normalize(config: SharpConfig) -> SharpConfig:
    if config.config_id is None:
        config.config_id = str(sharp.config.directory.config_dir)
    config.output_dir = as_absolute_Path(config.output_dir)
    config.shared_output_dir = as_absolute_Path(config.shared_output_dir)
    return config


def as_absolute_Path(path: str) -> Path:
    ppath = Path(path).expanduser()
    if ppath.is_absolute():
        out = ppath
    else:
        out = sharp.config.directory.config_dir / ppath
    return out.resolve()


def get_tasks_tuple(config: SharpConfig) -> Tuple[LuigiTask, ...]:
    tasks = config.get_tasks()
    try:
        iter(tasks)
        return tuple(tasks)
    except TypeError:
        return (tasks,)
