from typing import Optional, Tuple

from fklab.segments import Segment
from sharp.data.types.aliases import NumpyArray
from sharp.data.types.signal import Signal
from sharp.tasks.signal.util import fraction_to_index


class Slice:
    """
    Given a slicing range (given as fractions of total signal length), a full
    signal, and optional segments, calculates signal and segment extract.
    """

    def __init__(
        self,
        bounds: Tuple[float, float],
        signal: Signal,
        segments: Optional[Segment] = None,
    ):
        self._signal_full = signal
        self._segments_full = segments
        self._indices = fraction_to_index(signal, bounds)

    @property
    def signal(self) -> Signal:
        return self._signal_full[slice(*self._indices)]

    @property
    def segments(self) -> Segment:
        selected_segs = self._segments_full.intersection(self.time_range)
        offset_segs = selected_segs - self.time_range[0]
        return offset_segs

    @property
    def time_range(self) -> NumpyArray:
        """ (start, stop) times of the slice. """
        return self._indices / self._signal_full.fs
