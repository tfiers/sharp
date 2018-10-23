"""
Standard Python types, stored on disk.
"""

import toml

from sharp.data.files.base import FileTarget
from sharp.util import cached


class TextFile(FileTarget):
    extension = ".txt"

    @cached
    def read(self) -> str:
        with open(self) as f:
            value = f.read()
        return value

    def write(self, value: str):
        # 'w' clears file.
        with open(self, "w") as f:
            f.write(value)


class FloatFile(TextFile):
    """
    A file containing only a scalar number.
    """

    extension = ".float" + TextFile.extension

    @cached
    def read(self) -> float:
        return float(super().read())

    def write(self, value: float):
        super().write(str(value))


class DictFile(FileTarget):
    """
    A dictionary serialised to be human readable.
    """

    extension = ".toml"

    def read(self) -> dict:
        return toml.load(self.path_string)

    def write(self, obj: dict):
        with open(self, "w") as f:
            toml.dump(obj, f)
