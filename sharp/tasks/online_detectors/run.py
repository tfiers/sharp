import numpy as np
from sharp.datatypes.neuralnet import SharpRNN


def calc_SOTA_output_envelope(
    lfp: np.ndarray, ripple_channel: int
) -> np.ndarray:
    ...


def calc_RNN_output_envelope(lfp: np.ndarray, model: SharpRNN) -> np.ndarray:
    ...
