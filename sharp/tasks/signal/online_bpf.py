from logging import getLogger

import numpy as np
from numpy.core.multiarray import ndarray
from scipy.signal import cheb2ord, cheby2, lfilter

from sharp.data.types.signal import Signal
from sharp.data.files.numpy import SignalFile
from sharp.tasks.signal.base import EnvelopeMaker

log = getLogger(__name__)


class ApplyOnlineBPF(EnvelopeMaker):

    title = "Online band-pass filter"

    def output(self):
        self.output_dir
        return SignalFile(self.output_dir, filename="causal-BPF")

    def run(self):
        fs = self.reference_channel_full.fs
        b, a = get_SOTA_online_BPF(fs)
        filtered = lfilter(b, a, self.reference_channel_full.as_vector())
        envelope = np.abs(filtered)
        self.output().write(Signal(envelope, fs))


def get_SOTA_online_BPF(
    fs: float,
    left_edge: ndarray = (90, 110),
    right_edge: ndarray = (190, 210),
    passband_ripple: float = 1,
    #   Maximum attenuation in the passband, in dB.,
    attenuation: float = 40,
    #   Minimum attenuation of the stopband, in dB.
) -> (ndarray, ndarray):
    """
    State of the art online IIR band pass filter.

    :param fs: Signal sampling frequency.
    :param left_edge: Left edge of passband. In hertz.
    :param right_edge: idem for right edge.
    :param passband_ripple: Maximum attenuation in the passband, in dB.
    :param attenuation: Minimum attenuation of the stopband, in dB.
    :return: Numerator and denominator coefficients of the IIR filter.

    Notes
    -----
    Design method and parameters are the same as in the file:
     - falcon/tests/filters/iir_ripple_low_delay/matlab_design/iir_ripple_low_delay.filter

    ..from the private Kloostermanlab 'RealTime/falcon' repository.
    """
    f_nyq = fs / 2
    wp = np.array((left_edge[1], right_edge[0])) / f_nyq
    ws = np.array((left_edge[0], right_edge[1])) / f_nyq
    order, critical_freqs = cheb2ord(wp, ws, passband_ripple, attenuation)
    b, a = cheby2(order, attenuation, critical_freqs, "bandpass")
    log.info(f"Online BPF filter order: {order}")
    log.info(f"b: {b}")
    log.info(f"a: {a}")
    return b, a
