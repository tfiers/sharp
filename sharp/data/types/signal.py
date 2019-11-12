from typing import Iterable, Sequence, Optional

from numpy import array, asarray, diff, linspace, ndarray, stack

from fklab.segments import Segment
from sharp.util.misc import format_duration
from sharp.data.types.aliases import ArrayLike
from sharp.tasks.signal.util import time_to_index


class Signal(ndarray):
    # How to subclass ndarray:
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    time_axis = 0
    channel_axis = 1

    def __new__(
        cls, data: ArrayLike, fs: float, units: Optional[str] = None
    ) -> "Signal":
        '''
        :param data
        :param fs:  Signal sampling frequency, in hertz.
        :param units
        '''
        instance = asarray(data).view(cls)
        instance.fs = fs
        instance.units = units
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
        return format_duration(self.duration)

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

    def extract(self, segments: Segment) -> Iterable["Signal"]:
        """ Cut out segments from this signal. """
        for seg in segments:
            yield self.time_slice(*seg)

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
