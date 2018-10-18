from typing import Sequence

from luigi import DictParameter, FloatParameter
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from matplotlib.ticker import PercentFormatter
from numpy import argmax, median, percentile, array
from seaborn import set_hls_values

from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import Axes
from sharp.data.types.evaluation import ThresholdSweep
from sharp.tasks.plot.summary.base import MultiEnvelopeSummary

DISCRETE = dict(lw=2, marker=".", ms=10)
CONTINUOUS = dict(lw=4)


class PlotLatencyAndPR(MultiEnvelopeSummary):
    """
    Draws curves of precision-recall points, one point for each threshold, and
    one curve for each algorithm.
    """

    zoom_from: float = FloatParameter(0)
    # Precision and recall will be plotted in the ranges [zoom_from, 1].
    margin: float = FloatParameter(0.05)
    # Margin around the axes, as a percentage of (1 - zoom_from)
    line_kwargs: dict = DictParameter(DISCRETE)

    def output(self):
        filename = f"PR-and-latency--{self.zoom_from}"
        return FigureTarget(self.output_dir, filename)

    def run(self):
        fig, axes = subplots(
            nrows=2,
            ncols=2,
            figsize=1.1 * array([9.5, 9]),
            gridspec_kw=dict(width_ratios=[1.63, 1], height_ratios=[1, 1.63]),
        )  # type: Figure, Sequence[Axes]
        ax_PR = axes[1, 0]
        ax_delay_P = axes[1, 1]
        ax_delay_R = axes[0, 0]
        axes[0, 1].remove()
        self.setup_axes(ax_PR, ax_delay_P, ax_delay_R)
        self.plot_PR_curves(ax_PR)
        self.shade_under_PR_curves(ax_PR)
        self.plot_delays(ax_delay_P, ax_delay_R)
        fig.tight_layout()
        self.output().write(fig)

    def plot_PR_curves(self, ax: Axes):
        for sweep, title in zip(self.threshold_sweeps, self.titles):
            ax.plot(
                sweep.recall, sweep.precision, label=title, **self.line_kwargs
            )

    def shade_under_PR_curves(self, ax: Axes):
        """
        Fill area under each PR-curve with a light shade. Plot shades with
        highest AUC at the bottom (i.e. first, most hidden).
        """
        tups = zip(self.threshold_sweeps, self.colors)
        tups = sorted(tups, key=lambda tup: rank_higher_AUC_lower(tup[0]))
        for sweep, color in tups:
            fc = set_hls_values(color, l=0.9)
            ax.fill_between(sweep.recall, sweep.precision, color=fc)

    def plot_delays(self, ax_delay_P, ax_delay_R):
        for sweep, color in zip(self.threshold_sweeps, self.colors):
            rds = [te.rel_delays for te in sweep.threshold_evaluations]
            center = [median(rd) for rd in rds]
            low = [percentile(rd, 25) for rd in rds]
            high = [percentile(rd, 75) for rd in rds]
            b = argmax(sweep.recall > sweep.precision)  # Boundary
            kwargs = dict(color=color, lw=2)
            ax_delay_P.plot(center[b:], sweep.precision[b:], **kwargs)
            ax_delay_P.fill_betweenx(
                sweep.precision[b:], low[b:], high[b:], color=color, alpha=0.3
            )
            ax_delay_R.plot(sweep.recall[:b], center[:b], **kwargs)
            ax_delay_R.fill_between(
                sweep.recall[:b], low[:b], high[:b], color=color, alpha=0.3
            )

    def setup_axes(self, ax_PR: Axes, ax_delay_P: Axes, ax_delay_R: Axes):
        lims = (self.zoom_from - self.lim_offset, 1 + self.lim_offset)
        percentages = PercentFormatter(xmax=1, decimals=0)
        ax_PR.set_xlim(lims)
        ax_PR.set_ylim(lims)
        # ax_PR.set_aspect("equal")
        ax_PR.xaxis.set_major_formatter(percentages)
        ax_PR.yaxis.set_major_formatter(percentages)
        ax_PR.xaxis.tick_top()
        ax_PR.yaxis.tick_right()
        ax_delay_P.yaxis.set_major_formatter(percentages)
        ax_delay_P.xaxis.set_major_formatter(percentages)
        ax_delay_P.xaxis.set_label_position("top")
        ax_delay_P.xaxis.tick_top()
        ax_delay_P.set_ylim(lims)
        ax_delay_P.set_ylabel("Precision")
        ax_delay_P.set_xlabel("Detection latency")
        ax_delay_R.xaxis.set_major_formatter(percentages)
        ax_delay_R.yaxis.set_major_formatter(percentages)
        ax_delay_R.yaxis.set_label_position("right")
        ax_delay_R.yaxis.tick_right()
        ax_delay_R.set_xlim(lims)
        ax_delay_R.set_xlabel("Recall")
        ax_delay_R.set_ylabel("Detection latency")

    @property
    def lim_offset(self):
        return self.margin * (1 - self.zoom_from)


def rank_higher_AUC_lower(sweep: ThresholdSweep) -> float:
    return -sweep.AUC
