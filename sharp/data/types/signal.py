from typing import Sequence

import numpy as np

from fklab.segments import Segment
from sharp.data.types.aliases import ArrayLike, NumpyArray

# How to subclass np.ndarray:
# https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
#
from sharp.tasks.signal.util import time_to_index


class Signal(NumpyArray):
    def __new__(cls, input_array: ArrayLike, fs: float):
        """
        fs: signal sampling frequency, in hertz.
        """
        instance = np.asarray(input_array).view(cls)
        instance.fs = fs
        return instance

    # Make sure slices and views are also Signals.
    def __array_finalize__(self, instance):
        if instance is None:
            return
        else:
            self.fs = getattr(instance, "fs", None)

    @property
    def num_samples(self) -> int:
        return self.shape[0]

    @property
    def num_channels(self) -> int:
        if self.ndim > 1:
            return self.shape[1]
        else:
            return 1

    @property
    def duration(self) -> float:
        """ Length of signal, in seconds. """
        return self.num_samples / self.fs

    @property
    def duration_pretty(self) -> str:
        """ Length of signal, in human-readable format. """
        seconds = self.duration
        if seconds < 1:
            return f"{seconds * 1e3:.3g} ms"
        elif seconds < 60:
            return f"{seconds:.3g} seconds"
        elif seconds < 120:
            return f"1 minute, {seconds % 60:.1f} seconds"
        else:
            return f"{seconds // 60:g} minutes, {seconds % 60:.1f} seconds"

    def as_matrix(self) -> "Signal":
        if self.ndim == 1:
            return self[:, None]
        else:
            return self

    def as_vector(self) -> "Signal":
        if self.num_channels == 1:
            return self.flatten()
        else:
            raise ValueError("Signal has more than one channel.")

    @property
    def range(self) -> NumpyArray:
        """ (min, max) of signal values. """
        return np.array([self.min(), self.max()])

    @property
    def span(self) -> float:
        """ Difference between maximal and minimal signal value. """
        return np.diff(self.range)

    @property
    def time(self) -> NumpyArray:
        """ A time vector for this signal, starting at t = 0 seconds. """
        return self.get_time_vector(t0=0)

    def get_time_vector(self, t0: float) -> NumpyArray:
        """ A time vector for this signal, starting at t0. In seconds. """
        time = np.linspace(
            t0, t0 + self.duration, self.num_samples, endpoint=False
        )
        return time

    def extract(self, segments: Segment) -> Sequence["Signal"]:
        """ Cut out segments from this signal. """
        return [self.time_slice(*seg) for seg in segments]

    def time_slice(self, start: float, stop: float) -> "Signal":
        indices = time_to_index(
            [start, stop], self.fs, self.num_samples, clip=True
        )
        result = self.as_matrix()[slice(*indices), :]
        if self.ndim == 1:
            return result.as_vector()
        else:
            return result

    def fraction_to_index(self, fractions: ArrayLike) -> ArrayLike:
        """
        Convert fractions of total signal length to indices into this signal.
        """
        t = np.array(fractions) * self.duration
        return time_to_index(t, self.fs, self.num_samples, clip=True)


class BinarySignal(Signal):
    pass
