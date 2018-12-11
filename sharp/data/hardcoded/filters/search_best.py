from dataclasses import dataclass

from numpy.core.multiarray import array
from scipy.signal import butter, cheby1, cheby2, firwin, cheb2ord, iirfilter

from sharp.data.hardcoded.filters.base import (
    LTIRippleFilter,
    LTIRippleFilterFIR,
    HighpassLowpassCombi,
)


class OurButter(HighpassLowpassCombi):
    title = "Bread-n-butter"

    @property
    def tf_high(self):
        return butter(2, self.normalized_passband[0], "high")

    @property
    def tf_low(self):
        return butter(0, self.normalized_passband[1], "low")


class Butterworth(LTIRippleFilter):
    def tf(self, order, fs):
        return butter(order, self.normalized_passband, "bandpass")


@dataclass
class Cheby1(LTIRippleFilter):
    """
    max_passband_atten:  Maximum attenuation in the passband, in dB.
    """

    max_passband_atten: int = 4

    def tf(self, order, fs):
        return cheby1(
            order, self.max_passband_atten, self.normalized_passband, "bandpass"
        )


@dataclass
class Cheby2(LTIRippleFilter):
    """
    min_stopband_atten:  Minimum attenuation in the stopbands, in dB.
    """

    min_stopband_atten: int = 40

    def tf(self, order, fs):
        return cheby2(
            order, self.min_stopband_atten, self.normalized_passband, "bandpass"
        )


@dataclass
class WindowedSincFIR(LTIRippleFilterFIR):
    """
    Uses a Kaiser window (via SciPy's `kaiser_atten` and `kaiser_beta`
    functions).
    
    transition_width:  Desired width of the transition regions between
                passband and stopbands, as a fraction of the bandwidth.
    """

    transition_width: float = 0.1

    def b(self):
        tw = self.transition_width * self.bandwidth
        return firwin(
            self.num_taps(self.order),
            self.passband,
            width=tw,
            pass_zero=False,
            fs=self.fs,
        )


# kwargs for SearchLines_BPF
# --------------------------

main_comp = dict(
    filename="main-comp",
    filters={
        "Butterworth": Butterworth(),
        "Chebyshev Type I": Cheby1(),
        "Chebyshev Type II": Cheby2(),
        "Windowed Sinc FIR": WindowedSincFIR(),
    },
)

cheby1_comp = dict(
    filename="cheby1-comp",
    legend_title="Max. passband\nattenuation",
    sequential_colors=True,
    filters={
        "0.1 dB": Cheby1(max_passband_atten=0.1),
        "1 dB": Cheby1(max_passband_atten=1),
        "2 dB": Cheby1(max_passband_atten=2),
        "4 dB": Cheby1(max_passband_atten=4),
        "8 dB": Cheby1(max_passband_atten=8),
        "16 dB": Cheby1(max_passband_atten=16),
    },
)

cheby2_comp = dict(
    filename="cheby2-comp",
    legend_title="Min. stopband\nattenuation",
    sequential_colors=True,
    filters={
        "4 dB": Cheby2(min_stopband_atten=4),
        "10 dB": Cheby2(min_stopband_atten=10),
        "20 dB": Cheby2(min_stopband_atten=20),
        "40 dB": Cheby2(min_stopband_atten=40),
        "80 dB": Cheby2(min_stopband_atten=80),
        "120 dB": Cheby2(min_stopband_atten=120),
    },
)

sinc_FIR_comp = dict(
    filename="sinc-FIR-comp",
    legend_title="Transition\nwidth",
    sequential_colors=True,
    filters={
        "1 %": WindowedSincFIR(transition_width=0.01),
        "2 %": WindowedSincFIR(transition_width=0.02),
        "5 %": WindowedSincFIR(transition_width=0.05),
        "10 %": WindowedSincFIR(transition_width=0.10),
        "20 %": WindowedSincFIR(transition_width=0.20),
        "40 %": WindowedSincFIR(transition_width=0.40),
    },
)


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

    def tf(self, order, fs):
        """ `order` argument is ignored. """
        f_nyq = fs / 2
        wp = array((self.left_edge[1], self.right_edge[0])) / f_nyq
        ws = array((self.left_edge[0], self.right_edge[1])) / f_nyq
        order, critical_freqs = cheb2ord(
            wp, ws, self.passband_ripple, self.attenuation
        )
        return cheby2(order, self.attenuation, critical_freqs, "bandpass")

    passband = left_edge + right_edge


class FalconElliptic(LTIRippleFilter):
    """
    Based on
    https://bitbucket.org/kloostermannerflab/falcon/src/master/tests/filters/elliptic_rippleband.filter
    """

    def tf(self, order, fs):
        """ `order` argument is ignored. """
        return iirfilter(
            N=4, Wn=self.normalized_passband, rp=1, rs=80, ftype="ellip"
        )
