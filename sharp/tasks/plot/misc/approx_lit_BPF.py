from abc import abstractmethod

from matplotlib.axes import Axes
from numpy import (
    abs,
    angle,
    array,
    diff,
    linspace,
    log10,
    nan,
    ndarray,
    percentile,
    pi,
    unwrap,
    where,
)
from scipy.signal import (
    butter,
    cheb2ord,
    cheby2,
    firwin,
    freqs,
    freqz,
    savgol_filter,
    sosfreqz,
)

from sharp.config.load import final_output_dir
from sharp.config.style import paperfig
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.tasks.base import SharpTask
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.plot.util.legend import add_colored_legend


class PlotAllOnlineBPFReplications(SharpTask):
    def requires(self):
        return Falcon()
        # return (EgoStengel(), Dutta(), Falcon())


class OnlineBPFReplication(FigureMaker):

    fs_replication = 1000

    def output(self):
        filename = self.__class__.__name__
        return FigureTarget(final_output_dir / "approx-lit-BPF", filename)

    def work(self):
        fig, axes = subplots(
            nrows=2, ncols=2, figsize=paperfig(width=1.2, height=0.55)
        )
        ax_top_left: Axes = axes[0, 0]
        ax_top_right: Axes = axes[0, 1]
        ax_bottom_left: Axes = axes[1, 0]
        ax_bottom_right: Axes = axes[1, 1]
        ax_top_right.remove()
        ax_bottom_left.set_xlabel("Frequency (Hz)")
        ax_bottom_right.set_xlabel("Frequency (Hz)")
        ax_gain_dB = ax_top_left
        ax_gain = ax_bottom_left
        ax_grpdelay = ax_bottom_right
        ax_gain.set_ylabel("Gain")
        ax_gain_dB.set_ylabel("Gain (dB)")
        ax_gain_dB.set_ylim(-63, 4)
        ax_grpdelay.set_ylabel("Group delay (ms)")
        # Force zero-line in view:
        ax_grpdelay.axhline(y=0, color="none")
        for H in (self.H_original, self.H_replication):
            g = gain(H)
            ax_gain.plot(self.f, g)
            ax_gain_dB.plot(self.f, dB(g))
            ax_grpdelay.plot(self.f, group_delay(H, self.f))
        add_colored_legend(
            fig,
            (self.label_original, self.label_replication),
            loc="lower left",
            bbox_to_anchor=(0.5, 0.6),
        )
        fig.tight_layout(w_pad=3)
        self.output().write(fig)

    @property
    def label_original(self):
        return f"Original ($f_s =$ {self.fs_original:.0f})"

    @property
    def label_replication(self):
        return f"Replication ($f_s =$ {self.fs_replication:.0f})"

    def get_f_Nyq(self, fs=None):
        if fs is None:
            fs = self.fs_replication
        return fs / 2

    @property
    def f(self):
        """
        Frequencies (in Hz) at which the responses of a filter are calculated.
        """
        margin = 10  # To avoid phase discontinutities. In Hz.
        return linspace(margin, self.get_f_Nyq() - margin, 10000)

    def get_w(self, fs=None):
        """
        Normalized frequencies (in rad) at which the responses of a filter are
        calculated. For the returned values, pi ~ fs / 2.
        """
        return self.f / self.get_f_Nyq(fs) * pi

    def normalize(self, band, fs=None):
        """ [0, f_nyq] to [0, 1] """
        return array(band) / self.get_f_Nyq(fs)

    @property
    @abstractmethod
    def H_original(self):
        pass

    @property
    @abstractmethod
    def H_replication(self):
        pass


class EgoStengel(OnlineBPFReplication):

    band = (100, 400)

    @property
    def label_original(self):
        return "Original (continuous time)"

    @property
    def H_original(self):
        ba_high = butter(8, self.band[0], "high", analog=True)
        ba_low = butter(8, self.band[1], "low", analog=True)
        _, H_high = freqs(*ba_high, self.f)
        _, H_low = freqs(*ba_low, self.f)
        # Exact alternative:
        #   b = polymul(b_high, b_low)
        #   a = polymul(a_high, a_low)
        return H_high * H_low

    @property
    def H_replication(self):
        band = self.normalize(self.band)
        ba_high = butter(8, band[0], "high")
        ba_low = butter(2, band[1], "low")
        _, H_high = freqz(*ba_high, self.get_w())
        _, H_low = freqz(*ba_low, self.get_w())
        return H_high * H_low


class Dutta(OnlineBPFReplication):

    band = (150, 250)
    fs_original = 3000

    @property
    def H_original(self):
        fs = self.fs_original
        b = firwin(30, self.band, pass_zero=False, fs=fs)
        _, H = freqz(b, 1, self.get_w(fs))
        return H

    @property
    def H_replication(self):
        fs = self.fs_replication
        b = firwin(11, self.band, pass_zero=False, fs=fs)
        _, H = freqz(b, 1, self.get_w(fs))
        return H


# fmt: off
falcon_k = 0.009057795102643
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


class Falcon(OnlineBPFReplication):
    """
    Based on:
    https://bitbucket.org/kloostermannerflab/falcon/src/master/tests/filters/iir_ripple_low_delay/matlab_design/iir_ripple_low_delay.filter
    """

    # From comments; Seems wrong.
    # left_edge = (115, 135)
    # right_edge = (255, 275)

    left_edge = (125, 135)
    right_edge = (278, 300)
    max_passband_atten = 1  # dB
    min_stopband_atten = 40  # dB

    fs_original = 32000

    @property
    def H_original(self):
        fs = self.fs_original
        # Using b/a (transfer function) representation of filter results in
        # nonsense calculated frequency response. Second-order-sections (sos)
        # representation of filter avoids these numerical errors.
        _, H = sosfreqz(falcon_sos, self.get_w(fs))
        return falcon_k * H

    @property
    def H_replication(self):
        fs = self.fs_replication
        f_Nyq = fs / 2
        wp = array((self.left_edge[1], self.right_edge[0])) / f_Nyq
        ws = array((self.left_edge[0], self.right_edge[1])) / f_Nyq
        order, critical_freqs = cheb2ord(
            wp, ws, self.max_passband_atten, self.min_stopband_atten
        )
        ba = cheby2(order, self.min_stopband_atten, critical_freqs, "bandpass")
        _, H = freqz(*ba, self.get_w(fs))
        return H


def gain(H):
    """
    :param H:  Array of complex frequency responses of a filter.
    :return:  Gain of the filter (dimensionless).
    """
    return abs(H)


def dB(x):
    return 20 * log10(x)


def phase(H):
    """
    :return:  Phase of the filter, in radians.
    """
    return unwrap(angle(H))


def group_delay(H, f):
    """
    :param f:  Equi-spaced frequencies where H is calculated, in Hz.
    :return:  Group delays of the filter, in milliseconds.
    """
    df = diff(f[:2])
    phi = phase(H)
    tau = -savgol_filter(phi, 5, 3, deriv=1, delta=df)  # In seconds (1/Hz)
    mt = min(tau)
    remove_outliers(tau)
    return 1000 * tau


def remove_outliers(data: ndarray, m: float = 20):
    mag = abs(data)
    Q1, med, Q3 = percentile(mag, [25, 50, 75])
    threshold = med + m * (Q3 - Q1)
    outlier_ix = where(mag > threshold)
    data[outlier_ix] = nan
