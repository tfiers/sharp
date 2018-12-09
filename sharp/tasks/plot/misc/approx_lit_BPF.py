from matplotlib.axes import Axes
from numpy import abs, angle, array, diff, linspace, log10, pi, unwrap
from scipy.signal import butter, freqs, freqz, savgol_filter

from sharp.config.load import final_output_dir
from sharp.config.style import paperfig
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.tasks.plot.base import FigureMaker


class ApproximateLiteratureBPF(FigureMaker):

    fs_thesis = 1000
    band_thesis = (100, 200)

    def output(self):
        return FigureTarget(final_output_dir / "approx-lit-BPF", "ego-stengel")

    def work(self):
        fig, (ax_left, ax_right) = subplots(
            ncols=2, figsize=paperfig(height=0.4)
        )
        ax_gain: Axes = ax_left
        ax_tau: Axes = ax_right
        ax_gain.set_xlabel("Frequency (Hz)")
        ax_gain.set_ylabel("Gain (dB)")
        ax_gain.set_ylim(-90, 5)
        ax_tau.set_xlabel("Frequency (Hz)")
        ax_tau.set_ylabel("Group delay (ms)")
        for H in (self.H_study, self.H_thesis):
            ax_gain.plot(self.f, gain(H))
            ax_tau.plot(self.f, tau(H, self.f))
        fig.tight_layout()
        self.output().write(fig)

    @property
    def f_nyq(self):
        return self.fs_thesis / 2

    @property
    def f(self):
        margin = 10  # To avoid phase discontinutities. In Hz.
        return linspace(margin, self.f_nyq - margin, 10000)

    @property
    def w(self):
        return self.f / self.f_nyq * pi

    def normalize(self, band):
        return array(band) / self.f_nyq

    @property
    def H_study(self):
        band = (100, 400)
        ba_high = butter(8, band[0], "high", analog=True)
        ba_low = butter(8, band[1], "low", analog=True)
        _, H_high = freqs(*ba_high, self.f)
        _, H_low = freqs(*ba_low, self.f)
        return H_high * H_low

    @property
    def H_thesis(self):
        band = self.normalize(self.band_thesis)
        ba_high = butter(6, band[0], "high")
        ba_low = butter(3, band[1], "low")
        _, H_high = freqz(*ba_high, self.w)
        _, H_low = freqz(*ba_low, self.w)
        return H_high * H_low


def gain(H):
    """
    :param H:  Array of complex frequency responses of a filter.
    :return:  Gain of the filter, in dB.
    """
    return 20 * log10(abs(H))


def phase(H):
    """
    :return:  Phase of the filter, in radians.
    """
    return unwrap(angle(H))


def tau(H, f):
    """
    :param f:  Equi-spaced frequencies where H is calculated, in Hz.
    :return:  Group delays of the filter, in milliseconds.
    """
    phi = phase(H)
    df = diff(f[:2])
    tau = -savgol_filter(phi, 5, 3, deriv=1, delta=df)  # In seconds (1/Hz)
    return 1000 * tau
