from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from fklab.segments import Segment
from numpy import ndarray
from sharp.config import config, sharp_workflow
from sharp.OLD_datatypes.neuralnet import SharpRNN, torch_device
from sharp.OLD_datatypes.signal import Signal
from sharp.util.neuralnet import np_to_torch, to_batch


@dataclass
class TrainingPair:
    input_signal: torch.Tensor

    @property
    def desired_output_signal(self) -> torch.Tensor:
        return


def as_model_io(sig: Signal) -> torch.Tensor:
    """
    Convert a numpy signal to a pytorch array in the shape of a "batch" with
    one sample, and optionally moved to a GPU.

    input shape: (num_samples, num_channels)
    output shape: (1, num_samples, num_channels)
    """
    torch_array = np_to_torch(sig.as_matrix())
    batched = to_batch(torch_array, one_sample=True)
    return batched





@sharp_workflow.task
def train_RNN_one_epoch(
    lfp: ndarray, reference_SWRs: Segment, model: Optional[SharpRNN] = None
) -> SharpRNN:
    if model is None:
        # First epoch. Initialize a fresh model.
        model = SharpRNN(config.SharpRNN_config)


def calc_RNN_validation_performance(
    lfp: ndarray, reference_SWRs: Segment, model: SharpRNN
):
    ...


def select_RNN(
    model_validation_performances: Mapping[SharpRNN, float],
) -> SharpRNN:
    ...
