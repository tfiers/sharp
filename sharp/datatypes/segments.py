import fklab.segments
from fileflow import Saveable
from sharp.datatypes.base import ArrayFile


class SegmentArray(fklab.segments.Segment, Saveable):
    """ Make fklab's Segment saveable to disk, and give better name. """

    def saveable_as():
        return SegmentArrayFile


class SegmentArrayFile(ArrayFile):
    def write_to_file(self, seg: SegmentArray, f):
        super().write_to_file(seg.asarray(), f)

    def read_from_file(self, f) -> SegmentArray:
        array = super().read_from_file(f)
        return SegmentArray(data=array, check=False)
