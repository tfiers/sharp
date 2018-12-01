from logging import getLogger
from typing import List, Sequence

from matplotlib.axes import Axes

from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.data.types.style import fraction, paperfig
from sharp.tasks.evaluate.sweep import ThresholdSweeper
from sharp.tasks.plot.results.base import MultiEnvelopeFigureMaker
from sharp.tasks.plot.util.legend import add_colored_legend
from sharp.tasks.signal.base import EnvelopeMaker
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF

log = getLogger(__name__)


class PlotSearchLines(MultiEnvelopeFigureMaker):
    filename: str
    num_delays: Sequence[int]
    titles: List[str]
    envelope_maker_lists: Sequence[Sequence[EnvelopeMaker]]
    plot_IQR: bool = False

    subdir = "searcharrays"
    reference_maker = ApplyOnlineBPF()
    linestyle = ".-"
    reference_color = "silver"

    @property
    def envelope_makers(self):
        return [
            em for em_list in self.envelope_maker_lists for em in em_list
        ] + [self.reference_maker]

    @property
    def reference_sweep(self) -> ThresholdSweep:
        return get_sweep(self.reference_maker)

    @property
    def colors(self):
        return [f"C{i}" for i in range(len(self.titles))]

    def output(self):
        return FigureTarget(self.output_dir, self.filename)

    def work(self):
        fig, axes = subplots(nrows=2, sharex=True, figsize=paperfig())
        ax_top, ax_btm = axes
        ax_btm.set_xlabel("Number of delays")
        ax_F1 = ax_top
        ax_F1.set_ylabel("max $F_1$")
        ax_F1.yaxis.set_major_formatter(fraction)
        ax_delay = ax_btm
        ax_delay.set_ylabel("Latency")
        ax_delay.yaxis.set_major_formatter(fraction)
        self.plot_reference_F1(ax_F1)
        self.plot_reference_delay(ax_delay)
        tups = zip(self.envelope_maker_lists, self.colors, self.titles)
        for em_list, color, title in tups:
            log.info(f"Plotting searchline for {title}")
            sweeps = get_sweeps(em_list)
            self.plot_F1(ax_F1, sweeps, color)
            self.plot_delay(ax_delay, sweeps, color)
        fig.tight_layout()
        add_colored_legend(
            parent=ax_F1,
            labels=self.titles + ["Reference"],
            colors=self.colors + [self.reference_color],
        )
        self.output().write(fig)

    def plot_F1(self, ax: Axes, sweeps: Sequence[ThresholdSweep], color):
        F1s = [sweep.max_F1 for sweep in sweeps]
        ax.plot(self.num_delays, F1s, self.linestyle, color=color)

    def plot_delay(self, ax: Axes, sweeps: Sequence[ThresholdSweep], color):
        centers = [sweep.at_max_F1().rel_delays_median for sweep in sweeps]
        ax.plot(self.num_delays, centers, self.linestyle, color=color)
        if self.plot_IQR:
            low = [sweep.at_max_F1().rel_delays_Q1 for sweep in sweeps]
            high = [sweep.at_max_F1().rel_delays_Q3 for sweep in sweeps]
            ax.fill_between(self.num_delays, low, high, alpha=0.2, color=color)

    def plot_reference_F1(self, ax: Axes):
        ax.hlines(
            self.reference_sweep.max_F1,
            colors=self.reference_color,
            xmin=self.num_delays[0],
            xmax=self.num_delays[-1],
        )

    def plot_reference_delay(self, ax):
        ax.hlines(
            self.reference_sweep.at_max_F1().rel_delays_median,
            colors=self.reference_color,
            xmin=self.num_delays[0],
            xmax=self.num_delays[-1],
        )
        if self.plot_IQR:
            Q1 = self.reference_sweep.at_max_F1().rel_delays_Q1
            Q3 = self.reference_sweep.at_max_F1().rel_delays_Q3
            ax.fill_between(
                self.num_delays[[0, -1]],
                [Q1, Q1],
                [Q3, Q3],
                color=self.reference_color,
                alpha=0.2,
            )


def get_sweep(envelope_maker: EnvelopeMaker) -> ThresholdSweep:
    sweeper = ThresholdSweeper(envelope_maker=envelope_maker)
    sweep = sweeper.output().read()
    return sweep


def get_sweeps(
    envelope_makers: Sequence[EnvelopeMaker]
) -> Sequence[ThresholdSweep]:
    return [get_sweep(em) for em in envelope_makers]
