from typing import List, Sequence

from fklab.segments import Segment
from h5py import Dataset
from numpy import ndarray
from sharp.data.files.base import HDF5File
from sharp.util.misc import cached


class MultiChannelDataFile(HDF5File):
    """
    A list of arbitrarily sized and shaped NumPy arrays, stored on disk.
    """

    @cached
    def read(self) -> List[ndarray]:
        with self.open_file_for_read() as f:
            datasets: List[Dataset] = f.values()
            array_list = [None] * len(datasets)
            for dataset in datasets:
                full_path: str = dataset.name
                name = full_path.split("/")[-1]
                channel_nr = int(name)
                array_list[channel_nr] = dataset[()]
        return array_list

    def write(self, arrays: Sequence[ndarray]):
        with self.open_file_for_write() as f:
            for channel_nr, array in enumerate(arrays):
                f.create_dataset(name=str(channel_nr), data=array)


class MultiChannelSegmentsFile(MultiChannelDataFile):
    """
    A sequence of fklab Segments, stored on disk.
    (Where each "Segment" is actually a full sequence of (start, stop) tuples).
    """

    @cached
    def read(self) -> List[Segment]:
        return [Segment(array) for array in super().read()]

    def write(self, segs: Sequence[Segment]):
        super().write(seg.asarray() for seg in segs)
