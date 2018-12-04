from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy import array, diff, ndarray
from scipy.signal import butter, cheb2ord, cheby1, cheby2, firwin


def num_taps_BPF(order: int) -> int:
    return 2 * order + 1


def num_delays_BPF(order: int) -> int:
    return num_taps_BPF(order) - 1


class LTIRippleFilter(ABC):

    passband = (100, 200)

    @property
    def bandwidth(self):
        return diff(self.passband)

    @abstractmethod
    def get_taps(self, order, fs) -> (ndarray, ndarray):
        """
        :param order:  Order N of a typical band-pass filter, created by
                    convolution of a low-pass and a high-pass filter. (I.e.
                    order N for which num_taps = 2 * N + 1).
        :param fs:  Signal sampling frequency, in Hz.
        :return: (b, a), i.e. coefficients of (numerator, denominator) of the
        filter.
        """

    def get_passband_normalized(self, fs):
        f_nyq = fs / 2
        return array(self.passband) / f_nyq

    def __repr__(self):
        return str(self.__class__.__name__)


class Butterworth(LTIRippleFilter):
    def get_taps(self, order, fs):
        return butter(order, self.get_passband_normalized(fs), "bandpass")


@dataclass
class Cheby1(LTIRippleFilter):
    """
    max_passband_atten:  Maximum attenuation in the passband, in dB.
    """

    max_passband_atten: int = 4

    def get_taps(self, order, fs):
        return cheby1(
            order,
            self.max_passband_atten,
            self.get_passband_normalized(fs),
            "bandpass",
        )


@dataclass
class Cheby2(LTIRippleFilter):
    """
    min_stopband_atten:  Minimum attenuation in the stopbands, in dB.
    """

    min_stopband_atten: int = 40

    def get_taps(self, order, fs):
        return cheby2(
            order,
            self.min_stopband_atten,
            self.get_passband_normalized(fs),
            "bandpass",
        )


@dataclass
class WindowedSincFIR(LTIRippleFilter):
    """
    Uses a Kaiser window (via SciPy's `kaiser_atten` and `kaiser_beta`
    functions).
    
    transition_width:  Desired width of the transition regions between
                passband and stopbands, as a fraction of the bandwidth.
    """

    transition_width: float = 0.1

    def get_taps(self, order, fs):
        tw = self.transition_width * self.bandwidth
        b = firwin(
            num_taps_BPF(order), self.passband, tw, pass_zero=False, fs=fs
        )
        a = 1
        return b, a


class FalconCheby2(LTIRippleFilter):
    """
    "State-of-the-art" online IIR band pass filter.

    fs: Signal sampling frequency.
    left_edge: Left edge of passband. In hertz.
    right_edge: idem for right edge.
    passband_ripple: Maximum attenuation in the passband, in dB.
    attenuation: Minimum attenuation of the stopband, in dB.

    Notes
    -----
    Design method and parameters are the same as in the file:
     - falcon/tests/filters/iir_ripple_low_delay/matlab_design/iir_ripple_low_delay.filter
    from the private Kloostermanlab 'RealTime/falcon' repository.
    """

    left_edge = (90, 110)
    right_edge = (190, 210)
    passband_ripple = 1
    attenuation = 40

    def get_taps(self, order, fs):
        """ `order` argument is ignored. """
        f_nyq = fs / 2
        wp = array((self.left_edge[1], self.right_edge[0])) / f_nyq
        ws = array((self.left_edge[0], self.right_edge[1])) / f_nyq
        order, critical_freqs = cheb2ord(
            wp, ws, self.passband_ripple, self.attenuation
        )
        return cheby2(order, self.attenuation, critical_freqs, "bandpass")

    passband = left_edge + right_edge
