from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from h5py import Dataset

import fklab.segments
from fileflow import Saveable
from sharp.datatypes.base import HDF5File
from sharp.util.time import format_duration


@dataclass
class Signal(np.ndarray, Saveable):

    fs: float
    # Signal sampling frequency, in hertz.

    units: Optional[str] = None
    # Example: "μV"

    class Axes(IntEnum):
        TIME = 0
        CHANNEL = 1

    def get_filetype():
        return SignalFile

    # See "Subclassing ndarray":
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    def __new__(cls, data, fs: float, units: Optional[str] = None) -> "Signal":
        instance = super().__new__(cls, data)
        instance.fs = fs
        instance.units = units
        return instance

    # Make sure slices and views are also Signals.
    def __array_finalize__(self, instance):
        if instance is None:
            return
        else:
            self.fs = getattr(instance, "fs", None)
            self.units = getattr(instance, "units", None)

    @property
    def num_samples(self) -> int:
        return self.shape[self.Axes.TIME]

    @property
    def num_channels(self) -> int:
        if self.ndim > 1:
            return self.shape[self.Axes.CHANNEL]
        else:
            return 1

    @classmethod
    def from_channels(cls, channels: Sequence["Signal"]):
        data = np.stack([ch.as_vector() for ch in channels], axis=1)
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
    def range(self) -> np.ndarray:
        """ (min, max) of signal values. """
        return np.array([self.min(), self.max()])

    @property
    def span(self) -> float:
        """ Difference between maximal and minimal signal value. """
        return np.diff(self.range)

    @property
    def time(self) -> np.ndarray:
        """ A time vector for this signal, starting at t = 0 seconds. """
        return self.get_time_vector(t0=0)

    def get_time_vector(self, t0: float) -> np.ndarray:
        """ A time vector for this signal, starting at t0. In seconds. """
        time = np.linspace(
            t0, t0 + self.duration, self.num_samples, endpoint=False
        )
        return time

    def extract(self, segments: fklab.segments.Segment) -> Iterable["Signal"]:
        """ Cut out segments from this signal. """
        for seg in segments:
            yield self.time_slice(*seg)

    def time_slice(self, start: float, stop: float) -> "Signal":
        ix = self.index([start, stop])
        if self.ndim == 1:
            return self[slice(*ix)]
        else:
            return self[slice(*ix), :]

    def index(self, t) -> np.ndarray:
        """
        Convert times to array indices.

        :param t:  One or more times, in seconds. Number / array-like.
        :return:  A NumPy array of integer indices.
        """
        indices = (np.array(t) * self.fs).round().astype("int")
        return indices.clip(0, self.num_samples - 1)


class SignalFile(HDF5File[Signal]):
    def write_to_file(self, sig: Signal, f):
        f.create_dataset(self.main_key, data=sig.data)
        for k, v in asdict(sig).items():
            f.attrs[k] = v

    def read_from_file(self, f) -> Signal:
        dataset: Dataset = f[self.main_key]
        array = self.read_into_memory(dataset)
        return Signal(array, **dataset.attrs)
