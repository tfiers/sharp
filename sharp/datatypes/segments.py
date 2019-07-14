from dataclasses import dataclass
from typing import List

import numpy as np

import fklab.segments
from fileflow import Saveable
from sharp.datatypes.base import ArrayFile, MultichannelArrayFile


@dataclass
class SegmentsEventsIntersection:
    """
    Named & typed wrapper for "SegmentArray.contains()":
    """

    event_is_in_seg: np.ndarray
    # True for each event that is contained within any segment.

    num_events_in_seg: np.ndarray

    contains: np.ndarray
    # An N x 2 array, with for each segment 1..N, the start and end indices of
    # events that are contained within that segment.

    @property
    def index_of_first_event_in_seg(self) -> np.ndarray:
        """
        For each segment with at least one event in it, the index of the first
        such event.
        """
        return np.array(
            [first for (first, last) in self.contains if first != -1]
        )


class SegmentArray(fklab.segments.Segment, Saveable):
    """ Make fklab's Segment saveable to disk, and give better name. """

    def get_filetype():
        return SegmentArrayFile

    def contains(self, events: np.ndarray) -> SegmentsEventsIntersection:
        """ Segments and events are both assumed sorted. """
        return SegmentsEventsIntersection(super().contains(events))


class SegmentArrayFile(ArrayFile):
    def write_to_file(self, seg: SegmentArray, f):
        super().write_to_file(seg.asarray(), f)

    def read_from_file(self, f) -> SegmentArray:
        array = super().read_from_file(f)
        return SegmentArray(data=array, check=False)


class MultichannelSegmentArrayFile(MultichannelArrayFile):
    def write_to_file(self, arrays: List[SegmentArray], f):
        data = [seg.asarray() for seg in self]
        super().write_to_file(data, f)

    def read_from_file(self, f) -> List[SegmentArray]:
        return [
            SegmentArray(data=array, check=False)
            for array in super().read_from_file(f)
        ]
