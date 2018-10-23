from typing import Sequence

from numpy import array, asarray, diff, linspace, ndarray, stack

from fklab.segments import Segment
from sharp.data.types.aliases import ArrayLike
from sharp.tasks.signal.util import time_to_index


class Signal(ndarray):
    # How to subclass ndarray:
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    time_axis = 0
    channel_axis = 1

    def __new__(cls, input_array: ArrayLike, fs: float):
        """
        fs: signal sampling frequency, in hertz.
        """
        instance = asarray(input_array).view(cls)
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
        return self.shape[self.time_axis]

    @property
    def num_channels(self) -> int:
        if self.ndim > 1:
            return self.shape[self.channel_axis]
        else:
            return 1

    def to_channel_label(self, channel_index: int) -> str:
        """ Channel labels are a simple 1-based integer list, for now """
        return f"{channel_index + 1:.0f}"

    @classmethod
    def from_channels(cls, channels: Sequence["Signal"]):
        data = stack([ch.as_vector() for ch in channels], axis=1)
        return cls(data, channels[0].fs)

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
    def range(self) -> ndarray:
        """ (min, max) of signal values. """
        return array([self.min(), self.max()])

    @property
    def span(self) -> float:
        """ Difference between maximal and minimal signal value. """
        return diff(self.range)

    @property
    def time(self) -> ndarray:
        """ A time vector for this signal, starting at t = 0 seconds. """
        return self.get_time_vector(t0=0)

    def get_time_vector(self, t0: float) -> ndarray:
        """ A time vector for this signal, starting at t0. In seconds. """
        time = linspace(
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


class BinarySignal(Signal):
    pass
