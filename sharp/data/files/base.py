import os
from abc import ABC
from pathlib import PosixPath, WindowsPath
from typing import TypeVar
from h5py import File as HDF5File

from sharp.util.misc import make_parent_dirs


...
# We are not allowed to directly subclass pathlib.Path. (We therefore detect
# the OS and set the Path flavour manually).
if os.name == "nt":
    PathlibPath = WindowsPath
else:
    PathlibPath = PosixPath

T = TypeVar("T")


class FileTarget(PathlibPath, ABC):
    """
    Describes a file that is an output of one of the batch jobs in `tasks/`.
    """

    extension: str = ""
    # File extension, including leading dot. Subclasses should override at
    # class level.

    def write(self, object: T):
        """
        Write a value to the file.
        (Initial input files don't implement this).
        """

    def read(self) -> T:
        """
        Return the contents of the file.
        (Final output files don't implementthis).
        """

    def __new__(cls, file_path):
        make_parent_dirs(file_path)
        return PathlibPath.__new__(cls, file_path)

    @property
    def path_string(self):
        """ Path where this object is stored on disk, as a string. """
        return str(self)

    @property
    def size(self) -> str:
        """
        Giga and Mega are SI powers of 10 here -- not powers of 2 as Windows
        (wrongly) uses them (these are Gibi and Mebi).
        """
        size = self.stat().st_size  # In bytes
        for unit in ("bytes", "kB", "MB", "GB"):
            if size > 1000:
                size /= 1000
                continue
            else:
                break
        return f"{size:.1f} {unit}"


class HDF5Target(FileTarget, ABC):
    extension = ".hdf5"

    def open_file_for_read(self) -> HDF5File:
        """ Context manager for reading. """
        return HDF5File(self.path_string, "r")

    def open_file_for_write(self, clear_existing: bool = False) -> HDF5File:
        """ Context manager for writing. """
        if clear_existing:
            mode = "w"
        else:
            mode = "a"
        return HDF5File(self.path_string, mode)
