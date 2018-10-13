from typing import Tuple

from fklab.segments import Segment
from sharp.data.types.aliases import NumpyArray
from sharp.data.types.signal import Signal
from sharp.tasks.signal.util import fraction_to_index


class Slice:
    """
    Given full signals, calculates signal extracts (using a given range, given
    as fractions of total signal length).
    """

    def __init__(
        self,
        border_fractions: Tuple[float, float],
        input: Signal,
        envelope: Signal = None,
        reference_segs: Segment = None,
    ):
        self._range_indices = fraction_to_index(input, border_fractions)
        self._full_input = input
        self._full_envelope = envelope
        self._full_reference_segs = reference_segs

    @property
    def input(self) -> Signal:
        return self._full_input[slice(*self._range_indices)]

    @property
    def envelope(self) -> Signal:
        return self._full_envelope[slice(*self._range_indices)]

    @property
    def reference_segs(self) -> Segment:
        return self._full_reference_segs.intersection(self.range)

    @property
    def range(self) -> NumpyArray:
        """ (start, stop) times of the slice. """
        return self._range_indices / self._full_input.fs
