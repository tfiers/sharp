import numpy as np
from sharp.datatypes.neuralnet import SharpRNN


def calc_online_ripple_filter_envelope(
    LFP: np.ndarray, ripple_channel: int
) -> np.ndarray:
    ...


def calc_RNN_envelope(LFP: np.ndarray, model: SharpRNN) -> np.ndarray:
    ...
