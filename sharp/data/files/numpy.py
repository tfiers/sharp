"""
Describes how processed data is stored, and how it can be accessed.
"""
from logging import getLogger
from typing import List, Sequence

import numpy as np
from h5py import Dataset

from fklab.segments import Segment
from sharp.data.files.base import FileTarget, HDF5Target
from sharp.data.types.aliases import ArrayLike
from sharp.data.types.signal import Signal
from sharp.util.misc import cached


log = getLogger(__name__)


class NumpyArrayFile(FileTarget):
    """
    A numpy ndarray, stored on disk.
    """

    extension = ".npy"

    @cached
    def read(self) -> np.ndarray:
        # Might wanna try memmap here.
        return np.load(self.path_string)

    def write(self, array: ArrayLike):
        np.save(self.path_string, np.array(array))


class SignalFile(HDF5Target):
    """
    A `Signal` (subclass of a NumPy array), stored on disk.
    """

    KEY_SIG = "signal"
    KEY_FS = "sampling frequency (Hz)"
    KEY_UNITS = "units"

    @cached
    def read(self) -> Signal:
        log.info(f"Reading signal file at {self} ({self.size}) into memory.")
        with self.open_file_for_read() as f:
            dataset = f[self.KEY_SIG]
            array = dataset[:]
            fs = dataset.attrs[self.KEY_FS]
            units = dataset.attrs.get(self.KEY_UNITS, None)
        return Signal(array, fs, units)

    def write(self, signal: Signal):
        with self.open_file_for_write() as f:
            dataset = f.create_dataset(self.KEY_SIG, data=signal.data)
            dataset.attrs[self.KEY_FS] = signal.fs
            if signal.units:
                dataset.attrs[self.KEY_UNITS] = signal.units
        log.info(f"Wrote signal to {self} ({self.size})")


class MultiChannelDataFile(HDF5Target):
    """
    A sequence of arbitrarily sized and shaped NumPy arrays, stored on disk.
    """

    @cached
    def read(self) -> List[np.ndarray]:
        with self.open_file_for_read() as f:
            datasets: List[Dataset] = f.values()
            array_list = [None] * len(datasets)
            for dataset in datasets:
                channel_nr = int(dataset.name)
                array_list[channel_nr] = dataset[()]
        return array_list

    def write(self, arrays: Sequence[np.ndarray]):
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
