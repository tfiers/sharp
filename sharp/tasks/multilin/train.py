import numpy as np
from numpy import argmax, concatenate, cov, ndarray
from scipy.linalg import eigh

from sharp.config.load import intermediate_output_dir
from sharp.data.files.numpy import NumpyArrayFile
from sharp.data.types.signal import Signal
from sharp.tasks.base import SharpTask
from sharp.tasks.multilin.base import GEVecMixin
from sharp.tasks.signal.base import InputDataMixin
from sharp.util import compiled


class MaximiseSNR(SharpTask, InputDataMixin, GEVecMixin):
    def requires(self):
        return self.input_data_makers

    output_dir = intermediate_output_dir / "GEVecs"

    def output(self):
        return NumpyArrayFile(self.output_dir, self.filename)

    def run(self):
        signal = self.multichannel_train
        segments = self.reference_segs_train
        data = Signal(data=delay_stack(signal, self.delays), fs=signal.fs)
        reference = concatenate(data.extract(segments))
        background = concatenate(data.extract(segments.invert()))
        # Columns = channels = variables. Rows are (time) samples.
        Rss = _as_matrix(cov(reference, rowvar=False))
        Rnn = _as_matrix(cov(background, rowvar=False))
        GEVals, GEVecs = eigh(Rss, Rnn)
        first_GEVec = GEVecs[:, argmax(GEVals)]
        self.output().write(first_GEVec)


@compiled
def delay_stack(signal: ndarray, delays: ndarray):
    """
    At each multichannel time-sample, add copies of previous time samples as
    new 'channels'. For samples at the beginning, add multiple copies of the
    first sample, if necessary.

    :param signal:  array of shape (N, C)
    :param delays:  integer array of shape (d,)
    :return:  array of shape (N, d*C)
    """
    num_samples = signal.shape[0]
    num_channels = signal.shape[1]
    num_delays = delays.size
    output_shape = (num_samples, num_delays * num_channels)
    delay_stack = np.empty(output_shape)
    for sample in range(num_samples):
        col_start = 0
        col_end = num_channels
        for delay in delays:
            if sample - delay < 0:
                data = signal[0, :]
            else:
                data = signal[sample - delay, :]

            delay_stack[sample, col_start:col_end] = data
            col_start = col_end
            col_end += num_channels

    return delay_stack


def _as_matrix(array: ndarray):
    """
    :param array:  Either a scalar, or already a matrix
    :return:  A matrix
    """
    if array.ndim == 0:
        return array[None, None]
    else:
        return array
