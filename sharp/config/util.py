from pathlib import Path
from typing import Tuple
from warnings import warn

from typeguard import check_type

from sharp.config.spec import LuigiTask, SharpConfigBase, config_dir
from sharp.config.types import ConfigError


def validate(config: SharpConfigBase):
    settings = dir(SharpConfigBase)
    for name in dir(config):
        if name not in settings:
            warn(f"SharpConfig attribute `{name}` is not a config setting.")
        value = getattr(config, name)
        if value == NotImplemented:
            raise ConfigError(
                f"The mandatory config setting `{name}` is not set."
            )
        expected_type = config.__annotations__.get(name)
        if expected_type:
            try:
                check_type(name, value, expected_type)
            except TypeError as e:
                raise ConfigError(
                    f"The config setting `{name}` has an incorrect type. "
                    f"Expected type: `{expected_type}`. "
                    f"See preceding exception for details."
                ) from e


def normalize(config: SharpConfigBase):
    if config.config_id is None:
        config.config_id = str(config_dir)

    def as_absolute_Path(path: str) -> Path:
        ppath = Path(path).expanduser()
        if ppath.is_absolute():
            out = ppath
        else:
            out = config_dir / ppath
        return out.resolve()

    config.output_dir = as_absolute_Path(config.output_dir)
    config.shared_output_dir = as_absolute_Path(config.shared_output_dir)


def get_tasks_tuple(config: SharpConfigBase) -> Tuple[LuigiTask, ...]:
    tasks = config.get_tasks()
    try:
        iter(tasks)
        return tuple(tasks)
    except TypeError:
        return (tasks,)
