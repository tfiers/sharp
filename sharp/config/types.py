"""
Data types used in sharp/config/
As this file is at the top of the dependency graph for `sharp`, it should not
import from anywhere else in `sharp`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, TypeVar, Union

ConfigDict = Dict[str, Union[Any, "ConfigDict"]]

# We do not want to import from luigi yet wherever this type is used -- e.g. in
# config/spec.py. (This is because luigi executes initalization code at import
# time. We want to control this initialization by programmatically generating a
# luigi config file and setting env vars, later). Therefore make a dummy
# Luigi.Task type.
LuigiTask = TypeVar("Luigi.Task")

OneOrMoreLuigiTasks = Union[LuigiTask, Iterable[LuigiTask]]


@dataclass(frozen=True)
# Freeze to make class hashable (to be usable as a Luigi parameter)
class RecordingFileID:
    rat: int
    day: int
    probe: str
    path: Path

    @property
    def short_str(self) -> str:
        return f"rat_{self.rat}_day_{self.day}_{self.probe}"

    def __str__(self):
        return f"Rec(rat {self.rat}, day {self.day}, {self.probe})"


class ConfigError(Exception):
    """
    Raised when the environment is not configured properly to run `sharp`
    tasks.
    """
