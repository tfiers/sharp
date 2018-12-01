from logging import getLogger
from typing import Tuple

from luigi import IntParameter
from luigi.task import Parameter
from numpy import abs, array, ndarray
from scipy.signal import butter, cheb2ord, cheby1, cheby2, lfilter, firwin

from sharp.data.types.signal import Signal
from sharp.tasks.signal.base import EnvelopeMaker

log = getLogger(__name__)


passband = array([100, 200])


def critical_freqs(fs):
    f_nyq = fs / 2
    return passband / f_nyq


def butter_coeffs(N, fs):
    return butter(N, critical_freqs(fs), "bandpass")


def cheby1_coeffs(N, fs):
    max_passband_ripple = 4
    return cheby1(N, max_passband_ripple, critical_freqs(fs), "bandpass")


def cheby2_coeffs(N, fs):
    min_stopband_atten = 40
    return cheby1(N, min_stopband_atten, critical_freqs(fs), "bandpass")


def windowed_sinc_coeffs(N, fs):
    numtaps = N + 1
    # Use Kaiser window, via `kaiser_atten` and `kaiser_beta`.
    transition_width = 10  # Hz
    b = firwin(
        numtaps, passband, fs=fs, pass_zero=False, width=transition_width
    )
    a = 1
    return b, a


def get_SOTA_online_BPF(
    fs: float,
    left_edge: ndarray = (90, 110),
    right_edge: ndarray = (190, 210),
    passband_ripple: float = 1,
    attenuation: float = 40,
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
    wp = array((left_edge[1], right_edge[0])) / f_nyq
    ws = array((left_edge[0], right_edge[1])) / f_nyq
    order, critical_freqs = cheb2ord(wp, ws, passband_ripple, attenuation)
    b, a = cheby2(order, attenuation, critical_freqs, "bandpass")
    log.info(f"Online BPF filter order: {order}")
    # len(a) == 2 * order + 1
    return b, a


REFERENCE_FILTER = "falcon-cheby2"

online_BPFs = {
    "Butterworth": butter_coeffs,
    "Chebyshev Type I": cheby1_coeffs,
    "Chebyshev Type II": cheby2_coeffs,
    "Kaiser sinc": windowed_sinc_coeffs,
}


class ApplyOnlineBPF(EnvelopeMaker):

    filter_name = Parameter(default=REFERENCE_FILTER)
    N = IntParameter(default=None)
    # Filter order. numtaps = N + 1
    # (i.e. a 0-th order filter has 1 tap).

    title = "Single-channel BPF"

    output_subdir = "online-BPF"

    @property
    def output_filename(self):
        return f"{self.filter_name}, N={self.N}"

    def work(self):
        filtered = lfilter(*self.coeffs, self.input_signal)
        envelope = abs(filtered)
        envelope_sig = Signal(envelope, self.input_signal.fs)
        self.output().write(envelope_sig)

    @property
    def coeffs(self) -> Tuple[ndarray, ndarray]:
        """ Returns IIR (numer, denom), ie. (b, a) """
        if self.filter_name == REFERENCE_FILTER:
            coeffs_func = lambda N, fs: get_SOTA_online_BPF(fs)
        else:
            coeffs_func = online_BPFs[self.filter_name]
        return coeffs_func(self.N, self.input_signal.fs)

    @property
    def input_signal(self):
        return self.reference_channel_full.as_vector()
