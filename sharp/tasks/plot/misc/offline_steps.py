from colorsys import hls_to_rgb, rgb_to_hls
from logging import getLogger

from luigi import TupleParameter
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from numpy import abs, array, ceil, exp, imag, linspace, log, min, ndarray
from numpy.random import choice
from scipy.signal import hilbert
from sklearn.neighbors import KernelDensity

from fklab.signals.filter import apply_filter
from fklab.signals.smooth import smooth1d
from sharp.config.load import config
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.data.types.signal import Signal
from sharp.data.types.style import blue, green, paperfig, red
from sharp.tasks.base import SharpTask
from sharp.tasks.plot.base import FigureMaker, plot_signal_neat
from sharp.tasks.plot.util.scalebar import (
    add_voltage_scalebar,
    add_time_scalebar,
)
from sharp.tasks.signal.base import InputDataMixin
from sharp.util.misc import cached, ignore

logger = getLogger(__name__)


wideband_color = "black"
filter_output_color = green
Hilbert_color = "gray"
unsmoothed_envelope_color = blue
envelope_color = red
threshold_color = "black"
normal_lw = 1.5
thin_lw = 0.7 * normal_lw
bolder_lw = 1.25 * normal_lw
thicc_lw = 1.5 * normal_lw
annotation_text_size = 14


class PlotOfflineStepsMultifig(SharpTask):
    def requires(self):
        for time_range in config.offline_steps_segs:
            yield PlotOfflineSteps(time_range=time_range)


def add_scalebar(ax, label=False, y=0.01):
    kwargs = {}
    if not label:
        kwargs.update(label="")
    add_voltage_scalebar(ax, 250, "uV", pos_along=y, pos_across=0, **kwargs)


def add_title(ax: Axes, title, color, x=0.041, y=0.91, **kwargs):
    # Darken text color to compensate for visual effect where thin text looks
    # lighter than thick plot lines.
    ax.text(
        x=x,
        y=y,
        s=title,
        color=darken(color),
        transform=ax.transAxes,
        fontsize=annotation_text_size,
        **kwargs,
    )


def darken(color, amount: float = 0.14):
    input_tup = to_rgb(color)
    h, l, s = rgb_to_hls(*input_tup)
    l_darker = (1 - amount) * l
    output_tup = hls_to_rgb(h, l_darker, s)
    return output_tup


class PlotOfflineSteps(FigureMaker, InputDataMixin):

    time_range = TupleParameter()

    def output(self):
        start, stop = self.time_range
        return FigureTarget(
            directory=self.output_dir / "steps-offline-SWR-detection",
            filename=f"{start:.1f}--{stop:.1f}",
        )

    def requires(self):
        return self.input_data_makers

    def work(self):
        fig, axes = subplots(
            nrows=5,
            ncols=2,
            figsize=paperfig(1.2, 1.2),
            gridspec_kw=dict(
                width_ratios=(1, 0.26), height_ratios=(1, 1, 1, 1, 1.2)
            ),
        )
        self.remove_empty_axes(axes)
        self.plot_wideband(axes[0, 0])
        self.plot_filter_output(axes[1, 0])
        self.plot_analytic(axes[2, 0])
        self.plot_smoothed_envelope(axes[3, 0])
        self.plot_thresholded_envelope(axes[4, 0], axes[4, 1])
        with ignore(UserWarning):
            fig.tight_layout()
        self.output().write(fig)

    def remove_empty_axes(self, axes):
        for row in (0, 1, 2, 3):
            ax: Axes = axes[row, 1]
            ax.remove()

    def plot_wideband(self, ax):
        self.plot_signal(self.x_t, ax=ax, color=wideband_color)
        add_title(ax, "LFP recording $z_t$", wideband_color, y=0.89)
        add_scalebar(ax, label=True, y=0.5)
        add_time_scalebar(ax, 100, "ms", pos_along=0.72, pos_across=0.06)

    def plot_filter_output(self, ax):
        self.plot_signal(self.o_t, ax=ax, color=filter_output_color)
        add_title(
            ax, "Band-pass filter output $o_t$", filter_output_color, y=0.83
        )
        add_scalebar(ax, y=0.4)

    def plot_analytic(self, ax):
        self.plot_analytic_signal_components(ax)
        self.plot_analytic_inset(ax)
        add_title(ax, "Envelope $u_t$", unsmoothed_envelope_color, y=0.72)
        add_scalebar(ax, y=0.4)

    def plot_smoothed_envelope(self, ax):
        self.plot_signal(
            self.envelope_unsmoothed,
            ax,
            color=unsmoothed_envelope_color,
            lw=bolder_lw,
        )
        self.plot_signal(self.e_t, ax, color=envelope_color, lw=thicc_lw)
        add_title(ax, "Smoothed envelope $n_t$", envelope_color, y=0.7)
        add_scalebar(ax)

    def plot_thresholded_envelope(self, ax_main: Axes, ax_dist: Axes):
        # e_t, with segment bands, and summary stats to the right :)
        logger.info("Plotting thresholded envelope..")
        self.plot_signal(
            self.e_t, ax=ax_main, color=envelope_color, lw=thicc_lw
        )
        rm = self.reference_maker
        ax_main.hlines(rm.threshold_high, *self.time_range, lw=thin_lw)
        ax_main.hlines(rm.threshold_low, *self.time_range, lw=thin_lw)
        add_scalebar(ax_main)
        add_title(ax_main, "Thresholds", threshold_color, y=0.58)
        logger.info("Done")
        logger.info("Plotting envelope density..")
        self.plot_envelope_dist(ax_dist)
        logger.info("Done")
        ax_dist.set_ylim(ax_main.get_ylim())

    def plot_analytic_inset(self, ax_main: Axes):
        ax_inset: Axes = inset_axes(
            ax_main,
            width="100%",
            height="100%",
            bbox_to_anchor=(1.08, -0.16, 0.31, 1.4),
            bbox_transform=ax_main.transAxes,
        )
        self.plot_analytic_signal_components(
            ax_inset, lw_multiplier=1.5, clip_on=True
        )
        start, stop = self.time_range
        span = stop - start
        zoom_to = 0.88 + array([-0.03, +0.03])  # fractions of time_range
        ax_inset.set_xlim(start + zoom_to * span)
        add_title(ax_inset, "$H[o_t]$", Hilbert_color, x=0.15, y=0.2)

        class Corners:
            upper_right = 1
            upper_left = 2
            lower_left = 3
            lower_right = 4

        mark_inset(
            ax_main,
            ax_inset,
            Corners.upper_left,
            Corners.lower_left,
            fc="none",
            ec="0.5",
        )
        for spine in ax_inset.spines.values():
            spine.set_visible(True)

    def plot_analytic_signal_components(self, ax, lw_multiplier=1, **kwargs):
        self.plot_signal(
            imag(self.analytic),
            ax,
            color=Hilbert_color,
            lw=lw_multiplier * normal_lw,
            **kwargs,
        )
        self.plot_signal(
            self.o_t,
            ax,
            color=filter_output_color,
            lw=lw_multiplier * normal_lw,
            **kwargs,
        )
        self.plot_signal(
            self.envelope_unsmoothed,
            ax,
            color=unsmoothed_envelope_color,
            lw=lw_multiplier * bolder_lw,
            **kwargs,
        )

    def plot_envelope_dist(self, ax: Axes):
        sample = choice(self.e_t, 5000)
        kde = KernelDensity(bandwidth=0.02 * self.e_t.span)
        kde.fit(as_data_matrix(sample))
        e_dom = linspace(*self.e_t.range, 500)
        density = exp(kde.score_samples(as_data_matrix(e_dom)))
        ax.fill_betweenx(e_dom, density, color=envelope_color)
        rm = self.reference_maker
        kwargs = dict(xmin=0, xmax=1, color=threshold_color, lw=thin_lw)
        ax.axhline(rm.threshold_high, **kwargs)
        ax.axhline(rm.threshold_low, **kwargs)
        ax.axhline(rm.envelope_median, linestyle=":", **kwargs)
        ax.axhline(0, color="gray", lw=thin_lw)
        ax.set_xticks([])
        ax.set_yticks([])
        add_title(
            ax,
            "Empirical\ndistribution of $n_t$",
            envelope_color,
            x=0.1,
            y=0.8,
            clip_on=False,
        )
        text_kwargs = dict(
            x=1.05,
            color=threshold_color,
            transform=ax.get_yaxis_transform(),
            fontsize=0.75 * annotation_text_size,
            va="center",
        )
        ax.text(y=rm.threshold_high, s="$T_h$", **text_kwargs)
        ax.text(y=rm.threshold_low, s="$T_l$", **text_kwargs)
        ax.text(y=rm.envelope_median, s="Median", **text_kwargs)

    @property
    def x_t(self):
        return self.reference_channel_test

    @property
    def fs(self):
        return self.x_t.fs

    @property
    @cached
    def o_t(self):
        logger.info("Band-pass-filtering signal..")
        rm = self.reference_maker
        with ignore(FutureWarning):
            filter_output = apply_filter(
                self.x_t, rm.band, fs=self.fs, **rm.filter_options
            )
        logger.info("Done")
        return Signal(filter_output, self.fs)

    @property
    @cached
    def analytic(self):
        return hilbert_fast(self.o_t)

    @property
    @cached
    def envelope_unsmoothed(self):
        return abs(self.analytic)

    @property
    @cached
    def e_t(self):
        logger.info("Smoothing envelope..")
        unsmoothed = abs(self.analytic)
        rm = self.reference_maker
        with ignore(FutureWarning):
            envelope = smooth1d(
                self.envelope_unsmoothed, delta=1 / self.fs, **rm.smooth_options
            )
        logger.info("Done")
        return Signal(envelope, self.fs)

    def plot_signal(self, signal, ax, lw=normal_lw, **kwargs):
        plot_signal_neat(
            signal, self.time_range, ax, tight_ylims=True, lw=lw, **kwargs
        )


def hilbert_fast(sig: Signal):
    # Zero-pad signal to nearest power of 2 or 3 in order to speed up
    # computation
    powers = ceil(log(sig.num_samples) / log([2, 3]))
    N = int(min(array([2, 3]) ** powers))
    with ignore(FutureWarning):
        analytic = hilbert(sig, N)
    return Signal(analytic[: sig.num_samples], sig.fs)


def as_data_matrix(array: ndarray):
    return array[:, None]
