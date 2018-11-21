from luigi import TupleParameter
from numpy import abs, array, ceil, imag, log, min
from scipy.signal import hilbert

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

wideband_color = "black"
filter_output_color = green
Hilbert_color = "gray"
unsmoothed_envelope_color = blue
envelope_color = red
normal_lw = 1.5
medium_lw = 1.25 * normal_lw
thicc_lw = 1.5 * normal_lw


class PlotOfflineStepsMultiple(SharpTask):
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

    def output(self):
        start, stop = self.time_range
        return FigureTarget(
            directory=self.output_dir / "steps-offline-SWR-detection",
            filename=f"{start:.1f}--{stop:.1f}",
        )

    def requires(self):
        return self.input_data_makers

    def work(self):
        fig, axes = subplots(nrows=5, figsize=paperfig(0.9, 1.2))
        self.plot_wideband(axes[0])
        self.plot_filter_output(axes[1])
        self.plot_analytic(axes[2])
        self.plot_smoothed_envelope(axes[3])
        self.plot_thresholded_envelope(axes[4])
        fig.tight_layout()
        self.output().write(fig)

    def plot_wideband(self, ax):
        self.plot_signal(self.x_t, ax=ax, color=wideband_color)
        add_scalebar(ax, label=True, y=0.5)

    def plot_filter_output(self, ax):
        self.plot_signal(self.o_t, ax=ax, color=filter_output_color)
        add_scalebar(ax, y=0.4)

    def plot_analytic(self, ax):
        self.plot_signal(imag(self.analytic), ax, color=Hilbert_color)
        self.plot_signal(self.o_t, ax, color=filter_output_color)
        self.plot_signal(
            self.envelope_unsmoothed,
            ax,
            color=unsmoothed_envelope_color,
            lw=medium_lw,
        )
        add_scalebar(ax, y=0.4)

    def plot_smoothed_envelope(self, ax):
        self.plot_signal(
            self.envelope_unsmoothed,
            ax,
            color=unsmoothed_envelope_color,
            lw=medium_lw,
        )
        self.plot_signal(self.e_t, ax, color=envelope_color, lw=thicc_lw)
        add_scalebar(ax)

    def plot_thresholded_envelope(self, ax):
        # e_t, with segment bands, and summary stats to the right :)
        self.plot_signal(self.e_t, ax=ax, color=envelope_color, lw=thicc_lw)
        rm = self.reference_maker
        ax.hlines(rm.threshold_high, *self.time_range)
        ax.hlines(rm.threshold_low, *self.time_range, linestyles="dashed")
        add_scalebar(ax)

    @property
    def x_t(self):
        return self.reference_channel_test

    @property
    def fs(self):
        return self.x_t.fs

    @property
    @cached
    def o_t(self):
        rm = self.reference_maker
        with ignore(FutureWarning):
            filter_output = apply_filter(
                self.x_t, rm.band, fs=self.fs, **rm.filter_options
            )
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
        unsmoothed = abs(self.analytic)
        rm = self.reference_maker
        with ignore(FutureWarning):
            envelope = smooth1d(
                self.envelope_unsmoothed, delta=1 / self.fs, **rm.smooth_options
            )
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
