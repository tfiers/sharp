from logging import getLogger

from luigi import TupleParameter
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from numpy import abs, array, ceil, imag, log, min, linspace, ndarray, exp
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
from sharp.tasks.plot.util.scalebar import add_voltage_scalebar
from sharp.tasks.signal.base import InputDataMixin
from sharp.util.misc import cached, ignore

logger = getLogger(__name__)


wideband_color = "black"
filter_output_color = green
Hilbert_color = "gray"
unsmoothed_envelope_color = blue
envelope_color = red
normal_lw = 1.5
bolder_lw = 1.25 * normal_lw
thicc_lw = 1.5 * normal_lw


class PlotOfflineStepsMultifig(SharpTask):
    def requires(self):
        for time_range in config.offline_steps_segs:
            yield PlotOfflineSteps(time_range=time_range)


def add_scalebar(ax, label=False, y=0.01):
    kwargs = {}
    if not label:
        kwargs.update(label="")
    add_voltage_scalebar(ax, 250, "uV", pos_along=y, pos_across=0, **kwargs)


class PlotOfflineSteps(FigureMaker, InputDataMixin):

    time_range = TupleParameter()
    nsteps = 5

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
            nrows=self.nsteps,
            ncols=2,
            figsize=paperfig(1.2, 1.2),
            gridspec_kw=dict(width_ratios=(1, 0.3)),
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
        for row in range(self.nsteps - 1):
            ax: Axes = axes[row, 1]
            ax.remove()

    def plot_wideband(self, ax):
        self.plot_signal(self.x_t, ax=ax, color=wideband_color)
        add_scalebar(ax, label=True, y=0.5)

    def plot_filter_output(self, ax):
        self.plot_signal(self.o_t, ax=ax, color=filter_output_color)
        add_scalebar(ax, y=0.4)

    def plot_analytic(self, ax):
        self.plot_analytic_main(ax)
        self.plot_analytic_inset(ax)

    def plot_analytic_main(self, ax):
        self.plot_analytic_signal_components(ax)
        add_scalebar(ax, y=0.4)

    def plot_analytic_inset(self, ax_main: Axes):
        ax_inset: Axes = inset_axes(
            ax_main,
            width="100%",
            height="100%",
            bbox_to_anchor=(1, 0.25, 0.35, 1.4),
            bbox_transform=ax_main.transAxes,
        )
        self.plot_analytic_signal_components(
            ax_inset, lw_multiplier=1.5, clip_on=True
        )
        start, stop = self.time_range
        span = stop - start
        ax_inset.set_xlim(start + array([0.68, 0.75]) * span)

        class Corners:
            upper_right = 1
            upper_left = 2
            lower_left = 3
            lower_right = 4

        mark_inset(
            ax_main,
            ax_inset,
            Corners.upper_left,
            Corners.lower_right,
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

    def plot_smoothed_envelope(self, ax):
        self.plot_signal(
            self.envelope_unsmoothed,
            ax,
            color=unsmoothed_envelope_color,
            lw=bolder_lw,
        )
        self.plot_signal(self.e_t, ax, color=envelope_color, lw=thicc_lw)
        add_scalebar(ax)

    def plot_thresholded_envelope(self, ax_main: Axes, ax_dist: Axes):
        # e_t, with segment bands, and summary stats to the right :)
        self.plot_envelope_main(ax_main)
        self.plot_envelope_dist(ax_dist)
        ax_dist.set_ylim(ax_main.get_ylim())

    def plot_envelope_main(self, ax):
        self.plot_signal(self.e_t, ax=ax, color=envelope_color, lw=thicc_lw)
        rm = self.reference_maker
        ax.hlines(rm.threshold_high, *self.time_range)
        ax.hlines(rm.threshold_low, *self.time_range, linestyles="dashed")
        add_scalebar(ax)

    def plot_envelope_dist(self, ax):
        sample = choice(self.e_t, 10000)
        kde = KernelDensity(bandwidth=0.01 * self.e_t.span)
        kde.fit(as_data_matrix(sample))
        e_dom = linspace(*self.e_t.range, 1000)
        density = exp(kde.score_samples(as_data_matrix(e_dom)))
        ax.plot(density, e_dom, color=envelope_color, lw=thicc_lw)
        ax.fill_betweenx(e_dom, density, color=envelope_color, alpha=0.3)
        rm = self.reference_maker
        ax.axhline(rm.threshold_high, color="black")
        ax.axhline(rm.threshold_low, color="black", linestyle="dashed")
        # ax.grid("off")
        ax.set_xticks([])
        ax.set_yticks([])

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
