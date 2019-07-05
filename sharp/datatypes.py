""" Classes that describe how the data processed and produced by this package
is stored on disk and represented in memory.

The actual data is stored somewhere else on the file system (as specified in a
custom `config.py` file; see the `config/` directory).

Calculations performed in this module should be minimal (e.g property accesses
should return near-instantaneously).
"""
from abc import ABC
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, List, Optional, Sequence

import h5py
import numpy as np
import torch
import torch.nn

import fklab.segments
from farao import Saveable
from sharp.util.misc import format_duration
from sharp.util.signal import time_to_index


class SavedAsHDF5(Saveable, ABC):
    extension = ".hdf5"

    @staticmethod
    def file_reader(path) -> h5py.File:
        """ Context manager for reading. """
        return h5py.File(path, mode="r")

    @staticmethod
    def file_writer(path) -> h5py.File:
        """ Context manager for writing. """
        # Mode 'w' clears (truncates) the file if it already exists. 'a' leaves
        # existing contents intact.
        return h5py.File(path, mode="a")


class Array(np.ndarray, SavedAsHDF5):
    """ A numpy array saved as an HDF5 file. """

    # See "Subclassing ndarray":
    # https://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    DATASET_KEY = "array"

    def __new__(cls, input_array) -> "Array":
        return np.asarray(input_array).view(cls)

    def save(self, path):
        with self.file_writer(path) as f:
            f.create_dataset(self.DATASET_KEY, data=self)

    @classmethod
    def load(cls, path) -> "Array":
        with cls.file_reader(path) as f:
            data: np.ndarray = f[cls.DATASET_KEY][()]
        return data.view(cls)


class Signal(Array):
    @dataclass
    class AXES:
        TIME = 0
        CHANNEL = 1

    def __new__(cls, data, fs: float, units: Optional[str] = None) -> "Signal":
        """
        :param fs:  Signal sampling frequency, in hertz.
        """
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

    @dataclass
    class KEYS:
        SIGNAL = "signal"
        FS = "sampling frequency (Hz)"
        UNITS = "units"

    def save(self, path):
        with self.file_writer(path) as f:
            dataset = f.create_dataset(self.KEYS.SIGNAL, data=self.data)
            dataset.attrs[self.KEYS.FS] = self.fs
            if self.units:
                dataset.attrs[self.KEYS.UNITS] = self.units

    @classmethod
    def load(cls, path) -> "Signal":
        with cls.file_reader(path) as f:
            dataset = f[cls.KEYS.SIGNAL]
            array = dataset[()]
            fs = dataset.attrs[cls.KEYS.FS]
            units = dataset.attrs.get(cls.KEYS.UNITS, None)
        return Signal(array, fs, units)

    @property
    def num_samples(self) -> int:
        return self.shape[self.AXES.TIME]

    @property
    def num_channels(self) -> int:
        if self.ndim > 1:
            return self.shape[self.AXES.CHANNEL]
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
        indices = time_to_index(
            [start, stop], self.fs, self.num_samples, clip=True
        )
        result = self.as_matrix()[slice(*indices), :]
        if self.ndim == 1:
            return result.as_vector()
        else:
            return result


class ArrayList(List[np.ndarray], SavedAsHDF5):
    """
    An ordered collection of arbitrarily sized and shaped NumPy arrays, saved
    as an HDF5 file.
    """

    def save(self, path):
        with self.file_writer() as f:
            for index, array in enumerate(self):
                f.create_dataset(name=str(index), data=array)

    @classmethod
    def load(cls, path) -> "ArrayList":
        with cls.file_reader(path) as f:
            datasets: List[h5py.Dataset] = f.values()
            array_list = [None] * len(datasets)
            for dataset in datasets:
                full_path: str = dataset.name
                name = full_path.split("/")[-1]
                index = int(name)
                array_list[index] = dataset[()]
        return array_list


class Segment(fklab.segments.Segment, SavedAsHDF5):
    # Note: we shouldn't subclass Array (and thus ndarray), as fkklab's Segment
    # is not a subclass of ndarray (Liskov).

    def save(self, path):
        data = self.asarray()
        Array(data).save(path)

    @classmethod
    def load(cls, path):
        return cls(data=Array.load(path), check=False)


class SegmentList(List[fklab.segments.Segment], SavedAsHDF5):
    """
    A sequence of fklab Segments, stored on disk.
    (Where each "Segment" is actually a full sequence of (start, stop) tuples).
    """

    def save(self, path):
        data = [seg.asarray() for seg in self]
        ArrayList(data).save(path)

    @classmethod
    def load(cls, path):
        arrays = ArrayList.load(path)
        return cls([Segment(data=array, check=False) for array in arrays])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Memory location of PyTorch arrays (either at a GPU, or at the CPU).


@dataclass
class RNN(torch.nn.Module, SavedAsHDF5):
    """
    A recurrent neural network, with "gated recurrent units" (GRU) and a linear
    output layer (i.e. with unbounded output values).
    """

    num_input_channels: int
    num_layers: int = 2
    num_units_per_layer: int = 20
    dropout_probability: float = 0.4
    
    def __post_init__(self):
        super().__init__()
        if self.num_layers == 1:
            kwargs = dict()
        else:
            kwargs = dict(dropout=self.dropout_probability)
        self.gru = torch.nn.GRU(
            self.num_input_channels,
            self.num_units_per_layer,
            self.num_layers,
            batch_first=True,
            **kwargs,
        )
        self.h2o = torch.nn.Linear(
            in_features=self.num_units_per_layer, out_features=1
        )
    
    
    @dataclass
    class KEYS:
        STATE_DICT = "state_dict"
        NUM_INPUT_CHANNELS = "num_input_channels"
        NUM_LAYERS = "num_layers"
        NUM_UNITS_PER_LAYER = "num_units_per_layer"
        DROPOUT_PROBABILITY = "dropout_probability"


    def save(self, path):
        buffer = BytesIO()
        torch.save(self.state_dict(), buffer)
        with self.file_writer(path) as f:
            f
        
        
    def load(cls, path) -> "RNN":
        state_dict = torch.load(path, map_location=device)
        with cls.file_reader(path) as f:
            RNN()
        
    
    @property
    def device(self) -> torch.device:
        """ Assumes all parameters are on the same device. """
        return next(self.parameters()).device

    def get_init_h(self, num_batches: int = 1) -> torch.Tensor:
        """
        Random initial activations for the hidden units.
        """
        shape = (num_batches, self.gru.num_layers, self.gru.hidden_size)
        h = torch.randn(*shape)
        # Module.to() is in place, Tensor.to() is not (grr).
        return h.to(self.device)

    def forward(
        self, x: torch.Tensor, h0: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Calculate output values for one or more (multichannel) input signals.

        x: multichannel input array.
            Shape: (num_signals, num_time_steps, num_input_channels).
        h0: initial activations of all hidden layers.
            Shape: (num_signals, num_layers, num_units_per_layer).

        Returns:
            - Output channel (the activations of the last hidden layer, after
              an affine projection onto a line).
              Shape: (num_signals, num_time_steps, 1)
            - Activations of all hidden layers, at the last time-step.
              Shape: (num_signals, num_layers, num_units_per_layer).
        """
        # Somehow, the "batch_first" argument of GRU only applies to input (x)
        # & output (h_out), not to h_0 and h_n. Thus we permute dimensions
        # manually.
        h0 = h0.permute(1, 0, 2)
        # "h_out" are the activations of the last hidden layer, for all time
        # steps.
        # "h_n" are the activations of all hidden layers, for the last time
        # step.
        h_out, h_n = self.gru(x, h0)
        o = self.h2o(h_out)
        h_n = h_n.permute(1, 0, 2)
        return (o, h_n)
