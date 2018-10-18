from matplotlib.axes import Axes
from matplotlib.pyplot import figure
from matplotlib.ticker import PercentFormatter
from numpy import abs, argmin, ndarray
from pandas import DataFrame, concat
from seaborn import set_hls_values

from raincloud import distplot
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import Scalar
from sharp.data.types.evaluation import ThresholdSweep
from sharp.tasks.plot.summary.base import MultiEnvelopeSummary


class PlotLatency(MultiEnvelopeSummary):
    """
    Plots the distribution of relative detection delays, at a fixed recall
    value.
    """

    def output(self):
        return FigureTarget(self.output_dir, "Relative-delays")

    def run(self):
        fig = figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)  # type: Axes
        self.make_distplots(ax)
        self.format_xaxis(ax)
        # self.remove_text(ax)
        fig.tight_layout()
        self.output().write(fig)

    def format_xaxis(self, ax):
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    def remove_text(self, ax):
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticklabels([])
        ax.legend(None)

    def make_distplots(self, ax):
        delays = "Detection delay"
        precision = "Precision"
        recall = "Recall"
        algo = "Algorithm"
        df_dicts = (
            {
                delays: te.rel_delays,
                precision: te.precision,
                recall: te.recall,
                algo: title,
            }
            for sweep, title in zip(self.threshold_sweeps, self.titles)
            for te in select_threshold_evaluations(sweep)
        )
        df: DataFrame = concat(DataFrame(dic) for dic in df_dicts)
        if not df.empty:
            palette = [set_hls_values(c, l=0.6) for c in self.colors]
            distplot(
                data=df,
                x=delays,
                hue=algo,
                y=recall,
                ax=ax,
                bw=0.2,
                palette=palette,
                ms=6,
                move=0.2,
                offset=0.1,
            )


def select_threshold_evaluations(sweep: ThresholdSweep):
    choices = (
        (sweep.recall, 0.6),
        (sweep.recall, 0.8),
        (sweep.recall, 0.9),
        # (sweep.FDR, 0.1),
        # (sweep.FDR, 0.2),
        # (sweep.FDR, 0.4),
    )
    indices = [get_nearest_index(*c) for c in choices]
    evals = [sweep.threshold_evaluations[i] for i in indices]
    return evals


def get_nearest_index(haystack: ndarray, needle: Scalar):
    return argmin(abs(haystack - needle))
