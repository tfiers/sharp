"""
Describes how processed data is stored, and how it can be accessed.
"""

import numpy as np

from fklab.segments import Segment
from sharp.data.files.base import FileTarget
from sharp.data.files.stdlib import FloatFile
from sharp.data.types.aliases import ArrayLike
from sharp.data.types.signal import Signal
from sharp.util import cached


class NumpyArrayFile(FileTarget):
    """
    A numpy ndarray, stored on disk.
    """

    extension = ".npy"

    @cached
    def read(self) -> np.ndarray:
        # ToDo: try mmap
        return np.load(self.path_string)

    def write(self, array: ArrayLike):
        np.save(self.path_string, np.array(array))


class SignalFile(NumpyArrayFile):
    """
    A `Signal` (subclass of a NumPy array), stored on disk.
    We assume all signal files in a directory have the same sampling frequency.
    """

    @cached
    def read(self) -> Signal:
        array = super().read()
        fs = self._fs_file.read()
        return Signal(array, fs)

    def write(self, signal: Signal):
        super().write(signal)
        self._fs_file.write(signal.fs)

    @property
    def _fs_file(self):
        return FloatFile(self.parent, "sampling-frequency")


class SegmentsFile(NumpyArrayFile):
    """
    fklab Segments, stored on disk.
    """

    @cached
    def read(self) -> Segment:
        return Segment(super().read())

    def write(self, segs: Segment):
        super().write(segs.asarray())
