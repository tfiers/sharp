"""
Describes how processed data is stored, and how it can be accessed.
"""

import numpy as np

from fklab.segments import Segment
from sharp.data.files.base import FileTarget, HDF5Target
from sharp.data.types.aliases import ArrayLike
from sharp.data.types.signal import Signal
from sharp.util.misc import cached


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

    _key_sig = "signal"
    _key_fs = "sampling frequency (Hz)"

    @cached
    def read(self) -> Signal:
        with self.open_file_for_read() as f:
            dataset = f[self._key_sig]
            fs = dataset.attrs[self._key_fs]
        return Signal(dataset, fs)

    def write(self, signal: Signal):
        with self.open_file_for_write() as f:
            dataset = f.create_dataset(self._key_sig, data=signal.data)
            dataset.attrs[self._key_fs] = signal.fs


class SegmentsFile(NumpyArrayFile):
    """
    fklab Segments, stored on disk.
    """

    @cached
    def read(self) -> Segment:
        return Segment(super().read())

    def write(self, segs: Segment):
        super().write(segs.asarray())
