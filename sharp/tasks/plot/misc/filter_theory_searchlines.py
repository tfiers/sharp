from matplotlib.axes import Axes
from numpy import abs, any, linspace, log10, mean, median, nan, pi
from scipy.signal import BadCoefficients, freqz, group_delay, tf2zpk

from sharp.data.files.figure import FigureTarget
from sharp.tasks.plot.misc.searchlines import (
    BPF_SearchLines_Mixin,
    PlotSearchLines,
)
from sharp.tasks.signal.base import InputDataMixin
from sharp.util.misc import ignore

band = (100, 200)


class PlotFilterTheorySearchlines(
    PlotSearchLines, BPF_SearchLines_Mixin, InputDataMixin
):
    @property
    def output_dir(self):
        return super().output_dir / "filter-theory-searchlines"

    def requires(self):
        return self.input_data_makers

    def output(self):
        return FigureTarget(self.output_dir, self.filename)

    def plot_on_axes(self, ax_top: Axes, ax_btm: Axes):
        fs = self.reference_channel_full.fs
        ax_top.set_ylabel("Filter strength (dB)")
        ax_btm.set_ylabel("Median ripple-band\ngroup delay (ms)")
        for ripple_filter, color in zip(self.filters.values(), self.colors):
            bas = [ripple_filter.get_taps(order, fs) for order in self.orders]
            fss = [filter_strength(ba, fs) for ba in bas]
            ax_top.plot(self.num_delays, fss, self.linestyle, color=color)
            mgds = [median_group_delay(ba, fs) for ba in bas]
            ax_btm.plot(self.num_delays, mgds, self.linestyle, color=color)


def median_group_delay(ba, fs) -> float:
    """
    :return:  Median group delay of filter in `band`, in milliseconds. `nan` if
                filter poles lie so close to the unit circle that group delay
                cannot be reliably calculated.
    """
    f_nyq = fs / 2
    freqs = linspace(*band, num=1000) / f_nyq
    with ignore(BadCoefficients):
        _, p, _ = tf2zpk(*ba)
    if any(abs(p) > 0.97):
        return nan
    else:
        _, gd_samples = group_delay(ba, freqs)
        mgd_samples = median(gd_samples)
        mgd_seconds = mgd_samples / fs
        return mgd_seconds * 1000


dB = lambda x: 20 * log10(x)


def filter_strength(ba, fs) -> float:
    """
    :return:  Ratio of mean passband gain to mean stopband gain.
    
    We choose mean over median because "outliers" are important.
    """
    N_freqs = 1000
    with ignore(FutureWarning):
        w, h = freqz(*ba, N_freqs)
    f_nyq = fs / 2
    f = f_nyq * w / pi
    passband = (band[0] <= f) & (f < band[1])
    stopband = ~passband
    gain = abs(h)
    mg_passband = mean(gain[passband])
    mg_stopband = mean(gain[stopband])
    return dB(mg_passband / mg_stopband)
