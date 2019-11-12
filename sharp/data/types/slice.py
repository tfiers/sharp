from dataclasses import dataclass
from typing import Optional, Tuple

from numpy import ndarray, array

from fklab.segments import Segment
from sharp.data.types.signal import Signal


@dataclass
class Slice:
    """
    Given a slicing range (given as fractions of total signal length), a full
    signal, and optional segments, calculates signal and segment extract.
    """

    bounds: Tuple[float, float]
    signal_full: Signal
    segments_full: Optional[Segment] = None

    @property
    def time_range(self) -> ndarray:
        """ (start, stop) times of the slice. """
        return array(self.bounds) * self.signal_full.duration

    @property
    def signal(self) -> Signal:
        return self.signal_full.time_slice(*self.time_range)

    @property
    def segments(self) -> Segment:
        selected_segs = self.segments_full.intersection(self.time_range)
        offset_segs = selected_segs - self.time_range[0]
        return offset_segs
