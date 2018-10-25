import os
from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import PosixPath, WindowsPath
from typing import TypeVar, Union

from luigi import Target
from sharp.data.files.util import mkdir
from sharp.config.params import output_root

log = getLogger(__name__)


# We cannot directly subclass pathlib.Path (and we therefore have to detect OS
# manually).
if os.name == "nt":
    Path = WindowsPath
else:
    Path = PosixPath


T = TypeVar("T")


class FileTarget(Path, Target, ABC):
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

    def __new__(cls, directory: Union[Path, str], filename: str):
        dirr = mkdir(directory)
        path = dirr / (filename + cls.extension)
        return Path.__new__(cls, path)

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
