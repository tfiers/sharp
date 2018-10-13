from logging import getLogger
from typing import Callable, Iterable, Tuple

import numpy as np
import torch

from fklab.segments import Segment
from sharp.data.files.config import output_root
from sharp.data.types.aliases import TorchArray
from sharp.data.types.neuralnet import RNN
from sharp.data.types.signal import BinarySignal, Signal
from sharp.tasks.neuralnet.config import neural_net_config
from sharp.tasks.neuralnet.util import numpy_to_torch, to_batch
from sharp.tasks.signal.base import EnvelopeMaker
from sharp.tasks.signal.split import TrainTestSplitter
from sharp.tasks.signal.util import fraction_to_index, time_to_index

log = getLogger(__name__)

# (input_signal, target_signal)
IOTuple = Tuple[TorchArray, TorchArray]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Memory location of model (either at a GPU, or at the CPU).
log.info(f"Using {device} device.")


class NeuralNetTask(EnvelopeMaker):
    """
    Base class for other neural network tasks.
    """

    _model: RNN = None

    split_data = TrainTestSplitter()

    @property
    def output_dir(self):
        return output_root / "trained-networks"

    @property
    def target_signal(self) -> BinarySignal:
        """ Binary target, or training signal, as a one-column matrix. """
        segs = self.reference_segs.scale(
            1 + neural_net_config.reference_seg_extension, reference=1
        )
        return self.to_binary_signal(segs)

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
        return torch.nn.BCEWithLogitsLoss(reduction="sum")

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
            num_input_channels=1,
            num_layers=neural_net_config.num_layers,
            num_units_per_layer=neural_net_config.num_units_per_layer,
            p_dropout=neural_net_config.p_dropout,
        )
        # Module.to() is in place (Tensor.to() is not).
        model.to(device)
        return model

    def to_binary_signal(self, segs: Segment) -> BinarySignal:
        """
        Convert segment times to a binary signal (in a one-column matrix).
        """
        N = self.input_signal.shape[0]
        sig = np.zeros(N)
        for seg in segs:
            ix = time_to_index(seg, self.input_signal.fs, N, clip=True)
            sig[slice(*ix)] = 1
        return BinarySignal(sig, self.input_signal.fs).as_matrix()

    def make_io_tuple(self, fraction_seg: Tuple[float, float]) -> IOTuple:
        """
        Cut out the given segments from the input and target signal, and
        combines these as PyTorch-ready (input_slice, target_slice)-tuples.
        """
        for start, stop in fraction_to_index(self.input_signal, fraction_segs):
            yield (
                self.as_model_io(self.input_signal[start:stop]),
                self.as_model_io(self.target_signal[start:stop]),
            )

    def as_model_io(self, sig: Signal) -> TorchArray:
        """
        Convert a numpy array to a batched single-sample pytorch array,
        optionally moved to a GPU.

        input shape: (num_samples, num_channels)
        output shape: (1, num_samples, num_channels)
        """
        pytorch_array = numpy_to_torch(sig)
        batched = to_batch(pytorch_array, one_sample=True)
        return batched.to(device)
