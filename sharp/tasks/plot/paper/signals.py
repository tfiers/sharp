from matplotlib.axes import Axes
from numpy import ones, stack

from sharp.data.files.figure import FigureTarget
from sharp.data.hardcoded.style import paperfig, readable
from sharp.data.types.aliases import subplots
from sharp.data.types.signal import Signal
from sharp.data.types.split import TrainTestSplit
from sharp.tasks.plot.base import FigureMaker, plot_signal_neat
from sharp.tasks.plot.paper import output_dir, rm, sweepers, colors, get_tes
from sharp.tasks.plot.util.annotations import add_segments, add_event_arrows
from sharp.tasks.plot.util.scalebar import (
    add_time_scalebar,
    add_voltage_scalebar,
)

trange = [19.8, 20.8]
trange = [606.4, 607.5]
trange = [552.6, 553.2]  # early rnn ok visible. But one late SW
trange = [369.14, 369.65]  # three clean SPWR, early SW. But RNN not convincing
trange = [183.06, 184.5]
trange = [343.36, 344.3]  # nah, too weak SWs
trange = [183.09, 183.84]  # nice early RNN. Take me.


class PlotSignals(FigureMaker):
    def requires(self):
        return (rm,) + sweepers

    def output(self):
        return FigureTarget(output_dir, f"signals {trange[0]:.2f}")

    def work(self):
        nrows = 5
        axheights = ones(nrows)
        axheights[0] = 2
        axheights[1:3] = 0.84
        fig, axes = subplots(
            nrows=nrows,
            figsize=paperfig(0.57, 0.75),
            gridspec_kw=dict(height_ratios=axheights),
        )
        self.plot_input(axes[0])
        self.plot_offline(axes[1:3])
        self.plot_online(axes[3:])
        add_time_scalebar(axes[0], 200, in_layout=False, pos_along=0.56)
        fig.tight_layout()
        self.output().write(fig)

    def plot_input(self, ax):
        LFP_data = stack(
            [rm.sr_channel, rm.ripple_channel, rm.toppyr_channel], axis=1
        )
        LFP = Signal(LFP_data, rm.sr_channel.fs)
        plot_sig(LFP, ax)
        add_voltage_scalebar(ax)
        # add_segs(ax, rm.output().read())

    def plot_offline(self, axes):
        ax_SW = axes[0]
        ax_ripple = axes[1]
        plot_sig(rm.SW_envelope, ax_SW)
        plot_sig(rm.ripple_envelope, ax_ripple, tight_ylims=True)
        add_voltage_scalebar(ax_SW, pos_along=0.34)
        add_voltage_scalebar(ax_ripple, 100, pos_along=0.07)
        add_segs(ax_SW, rm.calc_SW_segments())
        add_segs(ax_ripple, rm.calc_ripple_segments())

    def plot_online(self, axes):
        for i, (sweeper, te, color) in enumerate(
            zip(sweepers, get_tes(), colors)
        ):
            ax: Axes = axes[i]
            plot_sig(sweeper.envelope_maker.envelope, ax, color=color)
            ax.hlines(
                te.threshold,
                *trange,
                linewidths=0.3 * readable["lines.linewidth"],
            )
            add_event_arrows(ax, te.correct_detections, color="green")
            add_event_arrows(ax, te.incorrect_detections, color="red")
            add_segs(ax, rm.output().read())


def plot_sig(sig, ax, **kwargs):
    plot_signal_neat(test_part(sig), trange, ax=ax, **kwargs)


def add_segs(ax, segs):
    add_segments(ax, test_part(rm.sr_channel, segs), color="grey")


def test_part(sig, segs=None):
    if segs is None:
        return TrainTestSplit(sig).signal_test
    else:
        return TrainTestSplit(sig, segs).segments_test
