import os
from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path, PosixPath, WindowsPath
from typing import TypeVar, Union

from luigi import Target
from sharp.config.load import output_root
from sharp.util.misc import cached

log = getLogger(__name__)


# We are not allowed to directly subclass pathlib.Path. (We therefore detect
# the OS and set the Path flavour manually).
if os.name == "nt":
    PathlibPath = WindowsPath
else:
    PathlibPath = PosixPath


T = TypeVar("T")


class FileTarget(PathlibPath, Target, ABC):
    """
    A file that is an output of one of the batch jobs in `../tasks`.

    A substitute for `luigi.LocalTarget`, with the user-friendliness of the
    Python `pathlib` module.
    """

    extension = ""
    # File extension, including leading dot. Subclasses should override at
    # class level.

    @abstractmethod
    def read(self) -> T:
        """ Return the contents of the file. """

    @abstractmethod
    def write(self, object: T):
        """ Write a value to the file. """

    @cached
    def __new__(cls, directory: Union[Path, str], filename: str):
        dirr = Path(directory)
        dirr.mkdir(parents=True, exist_ok=True)
        path = dirr / (filename + cls.extension)
        return PathlibPath.__new__(cls, path)

    @property
    def path_string(self) -> str:
        """ Path where this object is stored on disk, as a string. """
        return str(self)

    def delete(self):
        if self.exists():
            self.unlink()
            log.info(f"Deleted {self.relative_to(output_root)}")


class InputFileTarget(FileTarget, ABC):
    def write(self, object):
        """ Initial input files do not need to implement `write`. """


class OutputFileTarget(FileTarget, ABC):
    def read(self):
        """ Final output files do not need to implement `read`. """
