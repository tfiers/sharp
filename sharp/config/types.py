from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
# Freeze to make class hashable
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
