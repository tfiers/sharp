import os
from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path, PosixPath, WindowsPath
from typing import TypeVar, Union

from h5py import File as HDF5File
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
        # LaTeX includes can't have dots in filenames.
        filename_clean = filename.replace(".", "_")
        path = dirr / (filename_clean + cls.extension)
        return PathlibPath.__new__(cls, path)

    @property
    def path_string(self) -> str:
        """ Path where this object is stored on disk, as a string. """
        return str(self)

    def delete(self):
        if self.exists():
            self.unlink()
            log.info(f"Deleted {self.relative_to(output_root)}")

    @property
    def size(self) -> str:
        """
        Giga and Mega are SI multiples of 10 here, not powers of 2 as Windows
        (wrongly) uses them -- these are Gibi and Mebi.
        """
        size = self.stat().st_size  # In bytes
        for unit in ("bytes", "kB", "MB", "GB"):
            if size > 1000:
                size /= 1000
                continue
            else:
                break
        return f"{size:.1f} {unit}"


class InputFileTarget(FileTarget, ABC):
    def write(self, object):
        """ Initial input files do not need to implement `write`. """

    def delete(self):
        log.warning(f"Not deleting an external input file ({self})")


class OutputFileTarget(FileTarget, ABC):
    def read(self):
        """ Final output files do not need to implement `read`. """


class HDF5Target(FileTarget, ABC):
    extension = ".hdf5"

    def open_file_for_read(self) -> HDF5File:
        return HDF5File(self.path_string, "r")

    def open_file_for_write(self, clear_existing: bool = False) -> HDF5File:
        if clear_existing:
            mode = "w"
        else:
            mode = "a"
        return HDF5File(self.path_string, mode)
