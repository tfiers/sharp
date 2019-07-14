from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn
from fileflow import Saveable
from fklab.segments import Segment
from h5py import Group
from sharp.datatypes.base import HDF5File
from sharp.datatypes.signal import Signal
from sharp.util.neuralnet import np_to_torch, torch_device


class SharpRNN(torch.nn.Module, Saveable):
    """
    A recurrent neural network, with "gated recurrent units" (GRU) and a linear
    output layer (i.e. with unbounded output values).
    """

    def get_filetype():
        return SharpRNNFile

    @dataclass
    class Hyperparams:
        # Number of hidden layers
        num_layers: int = 2

        # Length of state vector of each hidden layer.
        num_units_per_layer: int = 20

        # For each element of the hidden state (except those of the last hidden
        # layer), probability that the element is forcibly zeroed during a
        # training step. This technique improves generalizability of the
        # trained network.
        dropout_probability: float = 0.4

        # The number of input channels is specific to each LFP recording. This
        # option thus needs not be set globally.
        num_input_channels: Optional[int] = None

    def __init__(self, config: Hyperparams):
        super().__init__()
        if config.num_input_channels is None:
            raise UserWarning(
                "Number of input channels must be set on model initialization."
            )
        if config.num_layers == 1:
            # Dropout is not applicable for one layer. Zero the dropout option
            # so that pytorch does not complain about this.
            config.dropout_probability = 0
        self.gru = torch.nn.GRU(
            config.num_input_channels,
            config.num_units_per_layer,
            config.num_layers,
            batch_first=True,
            dropout=config.dropout_probability,
        )
        self.h2o = torch.nn.Linear(
            in_features=self.num_units_per_layer, out_features=1
        )
        self.to(torch_device)

    def get_init_h(self, num_batches: int = 1) -> torch.Tensor:
        """
        Random initial activations for the hidden units.
        """
        shape = (num_batches, self.gru.num_layers, self.gru.hidden_size)
        h = torch.randn(*shape)
        # Module.to() is in place, Tensor.to() is not (grr).
        return h.to(torch_device)

    def forward(
        self, x: torch.Tensor, h0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate output values for one or more (multichannel) input signals.
        
        :param x:  Multichannel input array. Shape: (num_examples_in_batch,
                num_time_steps, num_input_channels).
        :param h0:  Initial activations of all hidden layers.
                Shape: (num_examples_in_batch, num_layers, num_units_per_layer).

        :return: A tuple (o, h_n):
            - o: Output array -- The activations of the last hidden layer, after
              an affine projection onto a line. Shape: (num_examples_in_batch,
              num_time_steps, 1)
            - h_n: Activations of all hidden layers, at the last time-step.
              Shape: (num_examples_in_batch, num_layers, num_units_per_layer).
        """
        # Somehow, the "batch_first" argument of GRU only applies to input (x)
        # & output (h_out), not to h_0 and h_n. Thus we permute dimensions
        # manually. "torch.transpose" is like "np.swapaxes".
        h0 = torch.transpose(h0, SharpRNN.IOAxes.TIME, SharpRNN.IOAxes.BATCH)
        # "h_out" are the activations of the last hidden layer, for all time
        # steps.
        # "h_n" are the activations of all hidden layers, for the last time
        # step.
        h_out, h_n = self.gru(x, h0)
        o = self.h2o(h_out)
        h_n = torch.transpose(h_n, SharpRNN.IOAxes.TIME, SharpRNN.IOAxes.BATCH)
        return (o, h_n)

    class IOAxes(IntEnum):
        """ Axes of tensors processed by a SharpRNN model """

        # Batch dimension. Each element is a batch sample/example/signal.
        BATCH = 0

        # Time dimension. Each element is a time sample/step.
        TIME = 1

        # Remaining dimension (e.g. probe channel, hidden layer
        # unit). Empty (size 1) for output tensor.
        OTHER = 2

    @staticmethod
    def as_batch(tensor: torch.Tensor, one_sample: bool = True) -> torch.Tensor:
        """
        PyTorch functions take data in "batches", i.e. with multiple "samples"
        at once. This function adds a size-one 'batch' dimension to the given
        array, if necessary.

        :param tensor
        :param one_sample:  Whether the given array consists of a single
                (training) example. For a SharpRNN model, one "sample"
                corresponds to one signal excerpt.
        """
        # "unsqueeze" adds an empty (size 1) dimension add the given index.
        if one_sample:
            return tensor.unsqueeze(SharpRNN.IOAxes.BATCH)
        else:
            if tensor.ndimension() == 1:
                # The array is a vector, with each element a different sample.
                return tensor.unsqueeze(SharpRNN.IOAxes.TIME)
            else:
                return tensor

    @staticmethod
    def to_IO_tensor(signal: Signal) -> torch.Tensor:
        """Convert NumPy signal to PyTorch tensor for use with this model.

        :return:  A PyTorch "batch" array with 1 sample.l
                Shape: (1, num_samples, num_channels)
        """
        torch_array = np_to_torch(signal.as_matrix())
        batched = SharpRNN.as_batch(torch_array, one_sample=True)
        return batched


class SharpRNNFile(HDF5File[SharpRNN]):
    def write_to_file(self, model: SharpRNN, f):
        group = f.create_group(self.main_key)
        for k, v in asdict(model).items():
            group.attrs[k] = v
        params_state = model.state_dict()
        for key in params_state:
            tensor: torch.Tensor = params_state[key]
            group.create_dataset(key, data=tensor.numpy())

    def read_from_file(self, f) -> SharpRNN:
        group: Group = f[self.main_key]
        model = SharpRNN(**group.attrs)
        params_state = {
            key: np_to_torch(self.read_into_memory(dataset))
            for key, dataset in group.items()
        }
        model.load_state_dict(params_state)


@dataclass
class Input_TargetPair:

    input_sig: Signal
    target_sig: Signal

    @property
    def torch_input(self):
        return SharpRNN.to_IO_tensor(self.input_sig)

    @property
    def torch_target(self):
        return SharpRNN.to_IO_tensor(self.target_sig)

    @staticmethod
    def create_from(
        input_signal: Signal, reference_SWRs: Segment
    ) -> "Input_TargetPair":
        """
        The created target is a one-channel array of the the same duration
        as "input_signal". It is 0 everywhere, except during reference SWR
        segments, where it is 1.
        """
        target_signal = Signal(
            np.zeros(input_signal.num_samples), input_signal.fs
        )
        for seg in reference_SWRs:
            start, stop = target_signal.index(seg)
            target_signal[start:stop] = 1
        return Input_TargetPair(input_signal, target_signal)
