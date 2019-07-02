from typing import List, Sequence

from fklab.segments import Segment
from sharp.data.files.base import ArrayFile, ArrayListFile
from sharp.util.misc import cached


class SegmentsFile(ArrayFile):
    def read(self) -> Segment:
        return Segment(data=super().read(), check=False)

    def write(self, seg: Segment):
        super().write(seg.asarray())


class MultiChannelSegmentsFile(ArrayListFile):
    """
    A sequence of fklab Segments, stored on disk.
    (Where each "Segment" is actually a full sequence of (start, stop) tuples).
    """

    @cached
    def read(self) -> List[Segment]:
        return [Segment(array) for array in super().read()]

    def write(self, segs: Sequence[Segment]):
        super().write(seg.asarray() for seg in segs)
