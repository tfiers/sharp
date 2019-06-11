from pathlib import Path
from typing import Union


def as_ID(text: str):
    """ Make a string usable as an Airflow task ID. """
    for forbidden_char in ", ()":
        text = text.replace(forbidden_char, "_")
    return text


def make_parent_dirs(file_path: Union[Path, str]):
    """
    Make sure the containing directories exist.

    :param file_path:  Pointing to a file in a directory.
    """
    dir_path: Path = resolve_path_str(file_path).parent
    dir_path.mkdir(exist_ok=True, parents=True)


def resolve_path_str(path: str) -> Path:
    return Path(path).expanduser().resolve().absolute()
