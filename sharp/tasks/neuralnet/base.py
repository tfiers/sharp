from logging import getLogger
from typing import Callable, Tuple

import numpy as np
import torch
from numpy import ndarray

from fklab.segments import Segment
from sharp.config.load import config, intermediate_output_dir
from sharp.data.types.aliases import TorchArray
from sharp.data.types.neuralnet import RNN
from sharp.data.types.signal import BinarySignal, Signal
from sharp.data.types.split import DataSplit
from sharp.tasks.neuralnet.util import numpy_to_torch, to_batch
from sharp.tasks.signal.base import InputDataMixin
from sharp.tasks.signal.util import time_to_index

log = getLogger(__name__)

# (input_signal, target_signal)
IOTuple = Tuple[TorchArray, TorchArray]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Memory location of model (either at a GPU, or at the CPU).
log.info(f"Using {device} device.")


class TrainValidSplit(DataSplit):
    split_fraction = 1 - config.valid_fraction

    @property
    def train_proper_slice(self):
        return self._left_slice

    @property
    def valid_slice(self):
        return self._right_slice


class NeuralNetMixin(InputDataMixin):
    """
    Base class for other neural network tasks.
    """

    output_dir = intermediate_output_dir / "trained-networks"

    _model: RNN = None

    @property
    def model(self) -> RNN:
        if self._model is None:
            self._model = self.get_model_prototype()
        return self._model

    @model.setter
    def model(self, value: RNN):
        self._model = value

    def get_model_prototype(self) -> RNN:
        """
        Initialises a new RNN with random weights.
        """
        model = RNN(
            num_input_channels=len(self.channels),
            num_layers=config.num_layers,
            num_units_per_layer=config.num_units_per_layer,
            p_dropout=config.p_dropout,
        )
        # Module.to() is in place (Tensor.to() is not).
        model.to(device)
        return model

    @property
    def channels(self):
        return config.channel_combinations[config.RNN_channel_combo_name]

    @property
    def cost_function(self) -> Callable[[TorchArray, TorchArray], TorchArray]:
        """
        Quantifies RNN performance by comparing network output with target
        signal (with lower values being better).

        Return a function that applies a sigmoid squashing function to its
        first argument, and that compares the result of this squashing with its
        second argument (a binary signal), by calculating the binary cross
        entropy between them.
        """
        return torch.nn.BCEWithLogitsLoss(
            reduction="sum", pos_weight=torch.Tensor(config.pos_weight)
        )

    @property
    def target_signal(self) -> BinarySignal:
        """ Binary target, or training signal, as a one-column matrix. """
        segs = self.reference_segs_train.scale(
            1 + config.reference_seg_extension, reference=1
        )
        N = self.reference_channel_train.shape[0]
        sig = np.zeros(N)
        # Convert segment times to a binary signal (in a one-column matrix) that
        # is as long as the full training input signal.
        if config.target_fullrect:
            self._add_rects(sig, segs)
        else:
            self._add_start_rects(sig, segs)
        # self._add_triangles(sig, segs)
        return Signal(sig, self.reference_channel_train.fs).as_matrix()

    def _add_rects(self, sig: ndarray, segs: Segment):
        for seg in segs:
            ix = time_to_index(
                seg, self.reference_channel_train.fs, sig.size, clip=True
            )
            sig[slice(*ix)] = 1

    def _add_triangles(self, sig: ndarray, segs: Segment):
        for seg in segs:
            ix = time_to_index(
                seg, self.reference_channel_train.fs, sig.size, clip=True
            )
            sig[slice(*ix)] = np.linspace(1, 0, np.diff(ix))

    def _add_start_rects(self, sig: ndarray, segs: Segment):
        for start, stop in segs:
            ix = time_to_index(
                [
                    start - config.target_start_pre,
                    start + config.target_start_post,
                ],
                self.reference_channel_train.fs,
                sig.size,
                clip=True,
            )
            sig[slice(*ix)] = 1

    @property
    def io_tuple_train(self) -> IOTuple:
        return (
            self.as_model_io(self.input_signal_train_proper),
            self.as_model_io(self.target_signal_train_proper),
        )

    @property
    def io_tuple_valid(self) -> IOTuple:
        return (
            self.as_model_io(self.input_signal_valid),
            self.as_model_io(self.target_signal_valid),
        )

    @property
    def input_signal_train_proper(self):
        return TrainValidSplit(
            self.multichannel_train[:, self.channels]
        ).train_proper_slice.signal

    @property
    def input_signal_valid(self):
        return TrainValidSplit(
            self.multichannel_train[:, self.channels]
        ).valid_slice.signal

    @property
    def target_signal_train_proper(self):
        return TrainValidSplit(self.target_signal).train_proper_slice.signal

    @property
    def target_signal_valid(self):
        return TrainValidSplit(self.target_signal).valid_slice.signal

    def as_model_io(self, sig: Signal) -> TorchArray:
        """
        Convert a numpy array to a batched single-sample pytorch array,
        optionally moved to a GPU.

        input shape: (num_samples, num_channels)
        output shape: (1, num_samples, num_channels)
        """
        pytorch_array = numpy_to_torch(sig.as_matrix())
        batched = to_batch(pytorch_array, one_sample=True)
        return batched.to(device)
