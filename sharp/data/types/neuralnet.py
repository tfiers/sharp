"""
Architecture of the recurrent neural network.
"""

import torch


class RNN(torch.nn.Module):
    """
    A recurrent neural network, with "gated recurrent units" (GRU) and a linear
    output layer (i.e. with unbounded output values).
    """

    def __init__(
        self,
        num_input_channels: int,
        num_layers: int = 2,
        num_units_per_layer: int = 20,
        p_dropout: float = 0.4,
    ):
        super().__init__()
        if num_layers == 1:
            kwargs = dict()
        else:
            kwargs = dict(dropout=p_dropout)
        self.gru = torch.nn.GRU(
            num_input_channels,
            num_units_per_layer,
            num_layers,
            batch_first=True,
            **kwargs,
        )
        self.h2o = torch.nn.Linear(
            in_features=num_units_per_layer, out_features=1
        )

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
        # Somehow, the `batch_first` argument of GRU only applies to input (x) &
        # output (h_out), not to h_0 and h_n. Thus we permute dimensions
        # manually.
        h0 = h0.permute(1, 0, 2)
        # `h_out` are the activations of the last hidden layer, for all time
        # steps.
        # `h_n` are the activations of all hidden layers, for the last time
        # step.
        h_out, h_n = self.gru(x, h0)
        o = self.h2o(h_out)
        h_n = h_n.permute(1, 0, 2)
        return (o, h_n)
