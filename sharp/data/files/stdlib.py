"""
Standard Python types, stored on disk.
"""

import pickle

from sharp.data.files.base import FileTarget
from sharp.util import cached


class FloatFile(FileTarget):
    """
    A file containing only a scalar number.
    """

    extension = ".float"

    @cached
    def read(self) -> float:
        with open(self) as f:
            value = f.read()
        return float(value)

    def write(self, value: float):
        # 'w' clears file.
        with open(self, "w") as f:
            f.write(str(value))


class PickleFile(FileTarget):
    """
    Serialization of (near) arbitrarily complex Python objects.
    """

    extension = ".pickle"

    @cached
    def read(self):
        with open(self, "rb") as f:
            value = pickle.load(f)
        return value

    def write(self, obj):
        with open(self, "wb") as f:
            pickle.dump(obj, f)
