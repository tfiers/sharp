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
from scipy.signal import butter, firwin, freqs, freqz, savgol_filter

from sharp.config.load import final_output_dir
from sharp.config.style import paperfig
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.tasks.base import SharpTask
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.plot.util.legend import add_colored_legend


class PlotAllOnlineBPFReplications(SharpTask):
    def requires(self):
        return (Dutta(), EgoStengel())


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
        ax_gain_dB.set_ylim(-43, 3)
        ax_grpdelay.set_ylabel("Group delay (ms)")
        ax_grpdelay.set_ylim(-3, 117)
        for H in (self.H_original, self.H_replication):
            g = gain(H)
            ax_gain.plot(self.f, g)
            ax_gain_dB.plot(self.f, dB(g))
            ax_grpdelay.plot(self.f, group_delay(H, self.f))
        add_colored_legend(
            fig,
            ("Original", "Replication"),
            loc="lower left",
            bbox_to_anchor=(0.5, 0.6),
        )
        fig.tight_layout()
        self.output().write(fig)

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
    def H_original(self):
        ba_high = butter(8, self.band[0], "high", analog=True)
        ba_low = butter(8, self.band[1], "low", analog=True)
        _, H_high = freqs(*ba_high, self.f)
        _, H_low = freqs(*ba_low, self.f)
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

    @property
    def H_original(self):
        fs = 3000
        b = firwin(30, self.band, pass_zero=False, fs=fs)
        _, H = freqz(b, 1, self.get_w(fs))
        return H

    @property
    def H_replication(self):
        fs = self.fs_replication
        b = firwin(11, self.band, pass_zero=False, fs=fs)
        _, H = freqz(b, 1, self.get_w(fs))
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
