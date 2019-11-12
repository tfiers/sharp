import numpy as np
from numpy import ndarray

from fklab.segments import Segment


class SegmentEventIntersection:
    """
    Named & typed wrapper for fklab's `Segment.contains()`:
    """

    def __init__(self, segs: Segment, events: ndarray):
        """ Segments are assumed sorted. """
        self._isinseg, self._ninseg, self._contains = segs.contains(events)
        # self._contains: An N x 2 array, with for each segment 1..N, the start
        # and end indices of events that are contained within that segment.

    @property
    def event_is_in_seg(self) -> ndarray:
        """ True for each event that is contained within any segment."""
        return self._isinseg

    @property
    def num_events_in_seg(self) -> ndarray:
        return self._ninseg

    @property
    def first_event_in_seg(self) -> ndarray:
        """
        For each segment with at least one event in it, the index of the first
        such event.
        """
        return np.array(
            [first for (first, last) in self._contains if first != -1]
        )
