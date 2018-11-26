from colorsys import hls_to_rgb, rgb_to_hls
from logging import getLogger

from luigi import TupleParameter
from matplotlib.axes import Axes
from matplotlib.collections import BrokenBarHCollection
from matplotlib.colors import to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from numpy import abs, array, ceil, diff, exp, imag, linspace, log, min, nonzero
from numpy.random import choice
from scipy.signal import hilbert
from sklearn.neighbors import KernelDensity

from fklab.signals.filter import apply_filter
from fklab.signals.smooth import smooth1d
from sharp.config.load import config
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.data.types.signal import Signal
from sharp.data.types.style import blue, green, paperfig, pink, red
from sharp.tasks.base import SharpTask
from sharp.tasks.plot.base import FigureMaker, plot_signal_neat
from sharp.tasks.plot.util.annotations import add_segments
from sharp.tasks.plot.util.scalebar import (
    add_time_scalebar,
    add_voltage_scalebar,
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
segment_color = pink
segment_alpha = 0.5
normal_lw = 1.5
thin_lw = 0.7 * normal_lw
bolder_lw = 1.25 * normal_lw
thicc_lw = 1.5 * normal_lw
annotation_text_size = 12


class PlotOfflineStepsMultifig(SharpTask):
    def requires(self):
        for time_range in config.offline_steps_segs:
            yield PlotOfflineSteps(time_range=time_range)


def add_scalebar(ax, label=False, y=0.01):
    kwargs = {}
    if not label:
        kwargs.update(label="")
    add_voltage_scalebar(ax, 250, "uV", pos_along=y, pos_across=0, **kwargs)


def add_title(ax: Axes, title, color, x=0.038, y=0.91, **kwargs):
    # Darken text color to compensate for visual effect where thin text looks
    # lighter than thick plot lines.
    ax.text(
        x=x,
        y=y,
        s=title,
        color=darken(color),
        transform=ax.transAxes,
        fontsize=annotation_text_size,
        clip_on=False,
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
            filename=f"{start:.1f}--{stop:.1f}".replace(".", "_"),
        )

    def requires(self):
        return self.input_data_makers

    def work(self):
        fig, axes = subplots(
            nrows=6,
            ncols=2,
            figsize=paperfig(1.2, 1.31),
            gridspec_kw=dict(
                width_ratios=(1, 0.24), height_ratios=(1, 1, 1, 1, 1.22, 0.8)
            ),
        )
        self.remove_empty_axes(axes)
        self.plot_wideband(axes[0, 0])
        self.plot_filter_output(axes[1, 0])
        self.plot_analytic(axes[2, 0])
        self.plot_smoothed_envelope(axes[3, 0])
        self.plot_thresholded_envelope(axes[4, 0], axes[4, 1])
        self.plot_segments(axes[5, 0])
        with ignore(UserWarning):
            fig.tight_layout()
        self.output().write(fig)

    def remove_empty_axes(self, axes):
        for row in (0, 1, 2, 3, 5):
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
            ax, "Band-pass filter\noutput $o_t$", filter_output_color, y=0.79
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
        add_title(ax, "Smoothed\nenvelope $n_t$", envelope_color, y=0.4)
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
        add_title(ax_main, "Thresholds $T$", threshold_color, y=0.58)
        segs = self.reference_segs_test
        visible_segs = segs.intersection(ax_main.get_xlim())
        bars = BrokenBarHCollection(
            xranges=[
                tup for tup in zip(visible_segs.start, visible_segs.duration)
            ],
            yrange=(0, rm.threshold_low),
            facecolors=segment_color,
            alpha=segment_alpha,
        )
        ax_main.add_collection(bars)
        # Find and plot crossings of lower threshold
        crossings_ix = nonzero(diff(self.e_t > rm.threshold_low))[0]
        crossings_t = crossings_ix / self.fs
        crossings_y = [rm.threshold_low] * len(crossings_t)
        ax_main.plot(crossings_t, crossings_y, ".", c="black")
        logger.info("Done")
        logger.info("Plotting envelope density..")
        self.plot_envelope_dist(ax_dist)
        logger.info("Done")
        ax_dist.set_ylim(ax_main.get_ylim())

    def plot_segments(self, ax):
        self.plot_signal(self.x_t, ax=ax, color="none", alpha=0.2, lw=thin_lw)
        # add_scalebar(ax, y=0.12)
        add_segments(
            ax,
            self.reference_segs_test,
            color=segment_color,
            alpha=segment_alpha,
        )
        add_title(
            ax,
            "Ripple\nsegments",
            segment_color,
            y=0.55,
            bbox=dict(facecolor="white", edgecolor="none"),
        )

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
        yrange = self.e_t.span
        sample = choice(self.e_t, 5000)
        kde = KernelDensity(bandwidth=0.02 * yrange)
        kde.fit(as_data_matrix(sample))
        e_dom = linspace(*self.e_t.range, 500)
        density = exp(kde.score_samples(as_data_matrix(e_dom)))
        ax.fill_betweenx(e_dom, density, color=envelope_color)
        ax.axhline(0, color="gray", lw=thin_lw)
        rm = self.reference_maker
        threshold_extent = 1.21
        kwargs = dict(
            xmin=0,
            xmax=threshold_extent,
            color=threshold_color,
            lw=thin_lw,
            clip_on=False,
        )
        ax.axhline(rm.threshold_high, **kwargs)
        ax.axhline(rm.threshold_low, **kwargs)
        ax.axhline(rm.envelope_median, linestyle=":", **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        add_title(
            ax, "Empirical\ndistribution of $n_t$", envelope_color, x=0.1, y=0.8
        )
        text_kwargs = dict(
            x=threshold_extent + 0.05,
            color=threshold_color,
            transform=ax.get_yaxis_transform(),
            fontsize=0.69 * annotation_text_size,
            va="center",
        )
        ax.text(y=rm.threshold_high, s="$T_{high}$", **text_kwargs)
        ax.text(y=rm.threshold_low, s="$T_{low}$", **text_kwargs)
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
    exponents = ceil(log(sig.num_samples) / log([2, 3]))
    N = int(min(array([2, 3]) ** exponents))
    with ignore(FutureWarning):
        analytic = hilbert(sig, N)
    return Signal(analytic[: sig.num_samples], sig.fs)


def as_data_matrix(array):
    return array[:, None]
