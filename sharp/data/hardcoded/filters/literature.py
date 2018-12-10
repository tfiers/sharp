from numpy import array, pi, polymul, tan
from scipy.signal import (
    butter,
    cheb2ord,
    cheby2,
    cont2discrete,
    firwin,
    normalize,
    sosfreqz,
)

from sharp.data.hardcoded.filters.base import (
    LTIRippleFilter,
    LTIRippleFilterFIR,
)


class EgoStengelFilter(LTIRippleFilter):
    passband = (100, 400)


class EgoStengelOriginal(EgoStengelFilter):
    fs = None

    @property
    def tf(self):
        b_high, a_high = butter(8, self.passband[0], "high", analog=True)
        b_low, a_low = butter(8, self.passband[1], "low", analog=True)
        b = polymul(b_high, b_low)
        a = polymul(a_high, a_low)
        return normalize(b, a)


class EgoStengelDiscretized(EgoStengelFilter):
    @property
    def tf(self):
        analog = EgoStengelOriginal()
        analog.passband = self.prewarped_passband
        b, a, _ = cont2discrete(analog.tf, self.dt, method="bilinear")
        # An idiosyncracy of cont2discrete:
        b = b.flatten()
        return b, a

    @property
    def prewarped_passband(self):
        T = self.dt
        return 2 / T * tan(array(self.passband) * T / 2)

    @property
    def dt(self):
        return 2 * pi / self.fs


class EgoStengelReplica(EgoStengelFilter):
    @property
    def tf(self):
        band = self.normalized_passband
        b_high, a_high = butter(8, band[0], "high")
        b_low, a_low = butter(2, band[1], "low")
        b = polymul(b_high, b_low)
        a = polymul(a_high, a_low)
        return normalize(b, a)


class DuttaFilter(LTIRippleFilterFIR):
    passband = (150, 250)
    numtaps: int = ...

    @property
    def b(self):
        return firwin(self.numtaps, self.passband, pass_zero=False, fs=self.fs)


class DuttaOriginal(DuttaFilter):
    numtaps = 30
    fs = 3000


class DuttaReplica(DuttaFilter):
    numtaps = 11


# Coefficients from: https://bitbucket.org/kloostermannerflab/falcon/src/master/tests/filters/iir_ripple_low_delay/matlab_design/iir_ripple_low_delay.filter

# fmt: off
falcon_gain = 0.009057795102643
falcon_sos = [
    [1, -1.999395559517321, 1, 1, -1.993909539451596, 0.996949008352066],
    [1, -1.996612125975065, 1, 1, -1.997891867240842, 0.998564128524551],
    [1, -1.999446113435119, 1, 1, -1.986445337112073, 0.989595554723684],
    [1, -1.996303149576368, 1, 1, -1.994638119350132, 0.995283281771532],
    [1, -1.999557218946594, 1, 1, -1.974166315776471, 0.977537474144240],
    [1, -1.995376455665212, 1, 1, -1.989909202085253, 0.990506963867645],
    [1, -1.999742604408150, 1, 1, -1.951752813122145, 0.955282256896558],
    [1, -1.992052662874018, 1, 1, -1.981463742801058, 0.982025842300048],
    [1, -1.999957242442239, 1, 1, -1.962151711669329, 0.962892139920814],
    [1, -1.952627879781272, 1, 1, -1.928156725555454, 0.930777461944991],
]
# fmt: on


class FalconOriginal(LTIRippleFilter):
    # Using b/a (transfer function) representation of this filter results in
    # nonsense calculated frequency response. Second-order-sections (sos)
    # representation of filter avoids these numerical errors.

    fs = 32000
    sos = falcon_sos
    gain = falcon_gain

    def tf(self):
        raise UserWarning("Please use `sos` and `gain` properties instead.")

    def freqresp(self, f):
        w = self.normalize_to_pi(f)
        _, H = sosfreqz(self.sos, w)
        return self.gain * H


class FalconReplica(LTIRippleFilter):
    # Design parameters also from the above URL.

    # Literally:
    # left_edge = (115, 135)
    # right_edge = (255, 275)
    max_passband_atten = 1  # dB
    min_stopband_atten = 40  # dB

    # Corrected:
    left_edge = (125, 135)
    right_edge = (278, 300)

    @property
    def tf(self):
        wp = array((self.left_edge[1], self.right_edge[0])) / self.f_Nyq
        ws = array((self.left_edge[0], self.right_edge[1])) / self.f_Nyq
        order, Wn = cheb2ord(
            wp, ws, self.max_passband_atten, self.min_stopband_atten
        )
        return cheby2(order, self.min_stopband_atten, Wn, "bandpass")
