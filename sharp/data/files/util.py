from pathlib import Path
from typing import Union


def mkdir(path: Union[Path, str]) -> Path:
    """ Return a directory that exists on the filesystem. """
    dirr = Path(path)
    dirr.mkdir(exist_ok=True, parents=True)
    return dirr
