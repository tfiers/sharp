from typing import Optional

import numba
import numpy as np

from sharp.data.files.numpy import SignalFile
from sharp.data.types.aliases import NumpyArray
from sharp.data.types.signal import Signal
from sharp.tasks.multilin.train import MaximiseSNR
from sharp.tasks.signal.base import EnvelopeMaker


class MultiChannelFilter(EnvelopeMaker):

    trainer = MaximiseSNR()

    def requires(self):
        return (self.trainer,) + super().requires()

    def output(self):
        return SignalFile(self.output_dir, "GEVec")

    def run(self):
        input_signal = self.input_signal_all.as_matrix()
        filter_weights = self.trainer.output().read()
        envelope = convolve_spatiotemporal(input_signal, filter_weights)
        self.output().write(Signal(envelope, input_signal.fs))


@numba.jit(nopython=True, cached=True)
def convolve_spatiotemporal(
    signal: NumpyArray, weights: NumpyArray, delays: Optional[NumpyArray] = None
) -> NumpyArray:
    """
    :param signal:  shape = (N, C)
    :param weights:  shape = (d*C,)
    :param delays:  shape = (d,). If None, assumes consecutive delays (i.e. no
                gaps)
    :return:  An array of shape (N,)
    """
    num_samples = signal.shape[0]
    num_channels = signal.shape[1]
    num_weights = weights.size
    if delays is None:
        delays = np.arange(num_weights // num_channels)

    num_delays = delays.size
    output = np.empty(num_samples)
    for sample in range(num_samples):
        y = 0
        for i, delay in enumerate(delays):
            if sample - delay < 0:
                signal = signal[0, :]
            else:
                signal = signal[sample - delay, :]

            w = weights[i * num_channels : (i + 1) * num_channels]
            y += np.sum(signal * w)

        output[sample] = y

    return output
