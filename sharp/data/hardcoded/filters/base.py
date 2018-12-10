from abc import ABC, abstractmethod
from typing import Optional

from numpy import array, diff, ndarray, pi
from scipy.signal import freqs, freqz


class LTIRippleFilter(ABC):
    """
    :param: order:  Order N of a typical band-pass filter, created by
                convolution of a low-pass and a high-pass filter. (I.e.
                order N for which num_taps = 2 * N + 1).
    :param: fs:  Signal sampling frequency, in Hz. `None` for continuous-time
                filters.
    """

    order: Optional[int] = None
    fs: Optional[float] = 1000
    passband = (100, 200)  # Hz

    @property
    @abstractmethod
    def tf(self) -> (ndarray, ndarray):
        """
        Transfer function.
        (b, a), i.e. coefficients of (numerator, denominator) of the filter.
        """

    def __repr__(self):
        return str(self.__class__.__name__)

    @property
    def bandwidth(self):
        return diff(self.passband)

    @property
    def f_Nyq(self):
        return self.fs / 2

    @property
    def normalized_passband(self):
        """ Scaling so that [0, f_nyq] --> [0, 1] """
        return array(self.passband) / self.f_Nyq

    def freqresp(self, f):
        """
        :param f:  Frequencies (in Hz) at which to calculate the output.
        :return:  Array H(f) of complex frequency responses of this filter.
        """
        if self.is_analog:
            calc_H = freqs
            w = f
        else:
            # For freqz, f == f_Nyq  //  w == pi
            # (and not w == 1, as in `iirfilter` e.g.)
            calc_H = freqz
            w = self.normalize_to_pi(f)
        _, H = calc_H(*self.tf, w)
        return H

    @property
    def is_analog(self) -> bool:
        return self.fs is None

    def normalize_to_pi(self, f):
        return pi * f / self.f_Nyq

    @staticmethod
    def num_taps(order: int) -> int:
        return 2 * order + 1

    @staticmethod
    def num_delays(order: int) -> int:
        return LTIRippleFilter.num_taps(order) - 1


class LTIRippleFilterFIR(LTIRippleFilter, ABC):
    @property
    def tf(self):
        return self.b, 1

    @property
    @abstractmethod
    def b(self):
        """ Coefficients of the convolution kernel. """
