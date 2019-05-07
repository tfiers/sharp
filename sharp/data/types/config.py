"""
Data types used in sharp/config/
As this file is at the top of the dependency DAG for `sharp`, it should not
import from anywhere else in `sharp`.
"""

from dataclasses import dataclass
from pathlib import Path
from textwrap import fill


@dataclass
class RecordingFile:
    rat: int
    day: int
    probe: str
    path: Path

    def __repr__(self):
        return f"RecordingFile(rat {self.rat}, day {self.day}, {self.probe})"


class ConfigError(Exception):
    """
    Raised when the environment is not configured properly to run `sharp`
    tasks.
    """

    def __init__(self, message: str):
        # Make sure the complete error message fits nice & square in the
        # terminal.
        future_prefix = f"{self.__class__}: "
        square_text = fill(future_prefix + message, width=80)
        square_text_with_prefix_sized_hole = square_text[len(future_prefix) :]
        super().__init__(square_text_with_prefix_sized_hole)
