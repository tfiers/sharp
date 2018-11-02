from typing import Tuple

from matplotlib.axes import Axes
from numpy import array
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.plot.results.base import MultiEnvelopeFigureMaker, fraction
from sharp.tasks.plot.style import blue, red
from sharp.tasks.plot.util.legend import add_colored_legend
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF


class PlotSearchArray(MultiEnvelopeFigureMaker):
    # num_delays = array(tuple(range(4)))
    num_delays = array(tuple(range(22)) + tuple(range(22, 40, 3)))
    linestyle = ".-"
    GEVec_color = blue
    sota_color = red

    convolvers = tuple(
        SpatiotemporalConvolution(num_delays=num) for num in num_delays
    )
    sota = ApplyOnlineBPF()
    envelope_makers = convolvers + (sota,)

    @property
    def convolver_sweeps(self) -> Tuple[ThresholdSweep, ...]:
        return self.threshold_sweeps[:-1]

    @property
    def sota_sweep(self) -> ThresholdSweep:
        return self.threshold_sweeps[-1]

    def output(self):
        return FigureTarget(self.output_dir, "searcharray")

    def work(self):
        fig, (top, btm) = subplots(nrows=2, sharex=True, figsize=(10, 9))
        self.plot_F1(top)
        self.plot_delay(btm)
        btm: Axes
        btm.set_xlabel("Number of delays")
        fig.tight_layout()
        add_colored_legend(
            parent=top,
            labels=("GEVec, all channels", self.sota.title),
            colors=(self.GEVec_color, self.sota_color),
        )
        self.output().write(fig)

    def plot_delay(self, ax: Axes):
        self.plot_sota_delay(ax)
        centers = [
            sweep.at_max_F1().rel_delays_median
            for sweep in self.convolver_sweeps
        ]
        low = [
            sweep.at_max_F1().rel_delays_Q1 for sweep in self.convolver_sweeps
        ]
        high = [
            sweep.at_max_F1().rel_delays_Q3 for sweep in self.convolver_sweeps
        ]
        ax.plot(
            self.num_delays, centers, self.linestyle, color=self.GEVec_color
        )
        ax.fill_between(
            self.num_delays, low, high, alpha=0.2, color=self.GEVec_color
        )
        ax.set_ylabel("Latency")
        # ax.invert_yaxis()
        ax.yaxis.set_major_formatter(fraction)

    def plot_sota_delay(self, ax):
        ax.hlines(
            self.sota_sweep.at_max_F1().rel_delays_median,
            colors=self.sota_color,
            xmin=self.num_delays[0],
            xmax=self.num_delays[-1],
        )
        Q1 = self.sota_sweep.at_max_F1().rel_delays_Q1
        Q3 = self.sota_sweep.at_max_F1().rel_delays_Q3
        ax.fill_between(
            self.num_delays[[0, -1]],
            [Q1, Q1],
            [Q3, Q3],
            color=self.sota_color,
            alpha=0.2,
        )

    def plot_F1(self, ax: Axes):
        F1s = [sweep.max_F1 for sweep in self.convolver_sweeps]
        ax.hlines(
            self.sota_sweep.max_F1,
            colors=self.sota_color,
            xmin=self.num_delays[0],
            xmax=self.num_delays[-1],
        )
        ax.plot(self.num_delays, F1s, self.linestyle, color=self.GEVec_color)
        # yticks = tuple(ax.get_yticks()) + (min(F1s), max(F1s))
        # ax.set_yticks(yticks)
        ax.set_ylabel("max $F_1$")
        ax.yaxis.set_major_formatter(fraction)
