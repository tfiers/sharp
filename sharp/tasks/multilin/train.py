import numpy as np
from luigi import IntParameter
from numpy import arange, argmax, cov
from numpy.core.multiarray import concatenate, ndarray
from scipy.linalg import eigh

from sharp.data.files.config import output_root
from sharp.data.files.numpy import NumpyArrayFile
from sharp.data.types.signal import Signal
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.base import InputDataMixin
from sharp.util import compiled


class MaximiseSNR(SharpTask, InputDataMixin):

    num_delays = IntParameter()

    def requires(self):
        return self.input_data_makers

    output_dir = output_root / "linear-filters"

    def output(self):
        return NumpyArrayFile(self.output_dir, "GEVec")

    def run(self):
        signal = self.input_signal_train.as_matrix()
        segments = self.reference_segs_train
        delays = arange(self.num_delays)
        if signal.num_channels == 1 and self.num_delays == 0:
            raise ValueError(
                "Need more than one channel or at least one delay to calc GEVec."
            )
        data = delay_stack(signal, delays)
        data_signal = Signal(data, signal.fs)
        reference = concatenate(data_signal.extract(segments))
        background = concatenate(data_signal.extract(segments.invert()))
        # Columns = channels = variables. Rows are (time) samples.
        Rss = cov(reference, rowvar=False)
        Rnn = cov(background, rowvar=False)
        GEVals, GEVecs = eigh(Rss, Rnn)
        first_GEVec = GEVecs[:, argmax(GEVals)]
        self.output().write(first_GEVec)


@compiled
def delay_stack(signal: ndarray, delays: ndarray):
    """
    At each time-sample, add new channels that are copies from previous time
    samples. For samples at the beginning, add multiple copies of the first
    sample, if necessary.

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