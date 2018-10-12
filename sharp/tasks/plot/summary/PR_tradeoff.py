import numpy as np
from luigi import BoolParameter, DictParameter, FloatParameter
from matplotlib.pyplot import figure
from matplotlib.ticker import PercentFormatter
from seaborn import set_hls_values

from sharp.data.types.aliases import Axes
from sharp.data.types.evaluation import ThresholdSweep
from sharp.data.files.figure import FigureTarget
from sharp.tasks.plot.summary.base import SummaryFigureMaker

DISCRETE = dict(lw=2, marker=".", ms=10)
CONTINUOUS = dict(lw=4)


class PlotPR(SummaryFigureMaker):
    """
    Draws curves of precision-recall points, one point for each threshold, and
    one curve for each algorithm.
    """

    start_proportion: float = FloatParameter(0)
    # Precision and recall will be plotted in the ranges [start_proportion, 1].

    margin: float = FloatParameter(0.05, significant=False)
    # Margin around the axes, as a percentage of (1 - start_proportion)

    line_kwargs: dict = DictParameter(CONTINUOUS)
    ticks_topright: bool = BoolParameter(False)
    figsize_multiplier: float = FloatParameter(1)

    def output(self):
        return FigureTarget(
            self.output_dir, f"PR-curve-{self.start_proportion}"
        )

    def run(self):
        fig = figure(figsize=self.figsize_multiplier * np.array((6.4, 6)))
        ax = fig.add_subplot(1, 1, 1)
        self.setup_FR_axes(ax)
        self.plot_curves(ax)
        self.plot_AUC(ax)
        self.mark_selected_recall_line(ax)
        fig.tight_layout()
        self.output().write(fig)

    def plot_curves(self, ax: Axes):
        for sweep, title in zip(self.evaluation.threshold_sweeps, self.titles):
            ax.plot(sweep.recall, sweep.FDR, label=title, **self.line_kwargs)

    def plot_AUC(self, ax: Axes):
        """
        Fill area under each PR-curve with a light shade. Plot shades with
        highest AUC at the bottom (first).
        """
        tups = zip(self.evaluation.threshold_sweeps, self.colors)
        tups = sorted(tups, key=lambda tup: rank_higher_AUC_lower(tup[0]))
        for sweep, color in tups:
            fc = set_hls_values(color, l=0.9)
            ax.fill_between(sweep.recall, sweep.FDR, 1, color=fc)

    def setup_PR_axes(self, ax: Axes):
        ax.set_aspect("equal")
        ax.set_xlim(
            self.start_proportion - self.lim_offset, 1 + self.lim_offset
        )
        ax.set_ylim(
            self.start_proportion - self.lim_offset, 1 + self.lim_offset
        )
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        if self.ticks_topright:
            ax.xaxis.tick_top()
            ax.yaxis.tick_right()

    def setup_FR_axes(self, ax: Axes):
        """
        Set up an axes to plot false discovery rate (1-precision) on the y-axis,
        and recall (= sensitivity) on the x-axis.
        """
        self.setup_PR_axes(ax)
        ax.set_ylim(
            -self.lim_offset, (1 - self.start_proportion) + self.lim_offset
        )
        ax.invert_yaxis()

    @property
    def lim_offset(self):
        return self.margin * (1 - self.start_proportion)

    def mark_selected_recall_line(self, ax: Axes):
        for sweep in self.evaluation.threshold_sweeps:
            recall = sweep.best.recall
            precision = sweep.best.precision
            ax.plot(
                recall,
                1 - precision,
                ".",
                ms=18,
                markeredgecolor="black",
                zorder=4,
            )
        # Background line to join both
        ax.axvline(x=recall, c="#555555", linestyle="dashed", lw=1.3, zorder=3)


def rank_higher_AUC_lower(sweep: ThresholdSweep) -> float:
    return -sweep.AUC
