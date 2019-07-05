from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn
from farao import Saveable
from h5py import Group
from sharp.datatypes.base import HDF5File
from sharp.util.neuralnet import torch_device, np_to_torch


class SharpRNN(torch.nn.Module, Saveable):
    """
    A recurrent neural network, with "gated recurrent units" (GRU) and a linear
    output layer (i.e. with unbounded output values).
    """

    @property
    def saveable_as():
        return SharpRNNFile

    @dataclass
    class Config:
        # Number of hidden layers
        num_layers: int = 2

        # Length of state vector of each hidden layer.
        num_units_per_layer: int = 20

        # For each element of the hidden state (except those of the last hidden
        # layer), probability that the element is forcibly zeroed during a training
        # step. This technique improves generalizability of the trained network.
        dropout_probability: float = 0.4

        # The number of input channels is specific to each lfp recording. This
        # option thus needs not be set globally.
        num_input_channels: Optional[int] = None

    def __init__(self, config: Config):
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


class SharpRNNFile(HDF5File[SharpRNN]):
    def write_to_file(self, model: SharpRNN, f):
        group = f.create_group(self.main_key)
        group.attrs = asdict(model)
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
