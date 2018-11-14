from numpy import abs, imag
from scipy.signal import hilbert

from fklab.signals.filter import apply_filter
from fklab.signals.smooth import smooth1d
from sharp.config.load import config
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.data.types.signal import Signal
from sharp.data.types.split import TrainTestSplit
from sharp.tasks.plot.base import FigureMaker, plot_signal_neat
from sharp.tasks.signal.base import InputDataMixin
from sharp.util.misc import cached, ignore


class PlotOfflineSteps(FigureMaker, InputDataMixin):

    trange = config.offline_steps_seg

    def output(self):
        return FigureTarget(self.output_dir, "offline-SWR-detection")

    def requires(self):
        return self.input_data_makers

    def work(self):
        fig, axes = subplots(nrows=5)
        self.plot_wideband(axes[0])
        self.plot_filter_output(axes[1])
        self.plot_analytic(axes[2])
        # e_t, with segment bands, and summary stats to the right :)
        self.plot_smoothed_envelope(axes[3])
        self.plot_thresholded_envelope(axes[4])
        fig.tight_layout()
        self.output().write(fig)

    def plot_wideband(self, ax):
        plot_signal_neat(self.x_t, self.trange, ax=ax)

    def plot_filter_output(self, ax):
        plot_signal_neat(self.o_t, self.trange, ax=ax)

    def plot_analytic(self, ax):
        plot_signal_neat(self.o_t, self.trange, ax, color="grey")
        plot_signal_neat(imag(self.analytic), self.trange, ax, color="grey")
        plot_signal_neat(abs(self.analytic), self.trange, ax)

    def plot_smoothed_envelope(self, ax):
        plot_signal_neat(abs(self.analytic), self.trange, ax, color="grey")
        plot_signal_neat(self.e_t, self.trange, ax)

    def plot_thresholded_envelope(self, ax):
        plot_signal_neat(self.e_t, self.trange, ax=ax, color="red")
        rm = self.reference_maker
        e_t_mario = TrainTestSplit(rm.envelope).signal_test
        plot_signal_neat(e_t_mario, self.trange, ax=ax)
        ax.hlines(rm.threshold_high, *self.trange)
        ax.hlines(rm.threshold_low, *self.trange, linestyles="dashed")

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
        analytic = hilbert(self.o_t)
        return Signal(analytic, self.fs)

    @property
    @cached
    def e_t(self):
        unsmoothed = abs(self.analytic)
        rm = self.reference_maker
        with ignore(FutureWarning):
            envelope = smooth1d(
                unsmoothed, delta=1 / self.fs, **rm.smooth_options
            )
        return Signal(envelope, self.fs)
