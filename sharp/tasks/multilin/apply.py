import numpy as np
from numba import prange
from numpy import ndarray

from sharp.data.types.signal import Signal
from sharp.tasks.multilin.base import GEVecMixin
from sharp.tasks.multilin.train import MaximiseSNR
from sharp.tasks.signal.base import EnvelopeMaker
from sharp.util.misc import compiled


class SpatiotemporalConvolution(EnvelopeMaker, GEVecMixin):

    output_subdir = "GEVec"

    @property
    def output_filename(self):
        return self.filename

    @property
    def title(self):
        return f"GEVec, {self.filename}"

    @property
    def trainer(self):
        return MaximiseSNR(
            num_delays=self.num_delays,
            channel_combo_name=self.channel_combo_name,
        )

    def requires(self):
        return (self.trainer,) + super().requires()

    def work(self):
        input_signal = self.multichannel_full[:, self.channels]
        filter_weights = self.trainer.output().read()
        filter_output = convolve_spatiotemporal(
            input_signal, filter_weights, self.delays
        )
        envelope = np.abs(filter_output)
        self.output().write(Signal(envelope, input_signal.fs))


@compiled
def convolve_spatiotemporal(
    signal: ndarray, weights: ndarray, delays: ndarray
) -> ndarray:
    """
    Implicitly stack delayed channels while convolving signal with weights.

    :param signal:  shape = (N, C)
    :param weights:  shape = (d*C,)
    :param delays:  shape = (d,)
    :return:  An array of shape (N,)
    """
    num_samples = signal.shape[0]
    num_channels = signal.shape[1]
    num_weights = weights.size
    num_delays = delays.size
    output = np.empty(num_samples)
    for sample in prange(num_samples):
        y = 0
        for i, delay in enumerate(delays):
            if sample - delay < 0:
                data = signal[0, :]
            else:
                data = signal[sample - delay, :]

            w = weights[i * num_channels : (i + 1) * num_channels]
            y += np.sum(data * w)

        output[sample] = y

    return output
