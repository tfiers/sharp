"""
Data types used in sharp/config/
As this file is at the top of the dependency graph for `sharp`, it should not
import from anywhere else in `sharp`.
"""

from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Any, Dict, TypeVar, Union, Iterable

ConfigDict = Dict[str, Union[Any, "ConfigDict"]]

# We do not want to import from luigi yet in config/spec.py. (As luigi executes
# initalization code on import. We want to control this initialization by
# setting env vars, later). Therefore make a dummy Luigi.Task type.
LuigiTask = TypeVar("LuigiTask")

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
        return f"rat {self.rat}, day {self.day}, {self.probe}, {self.path.name}"


class ConfigError(Exception):
    """
    Raised when the environment is not configured properly to run `sharp`
    tasks.
    """
