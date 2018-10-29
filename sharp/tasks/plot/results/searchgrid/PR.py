from matplotlib.axes import Axes

from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.tasks.plot.results.searchgrid.base import SearchGrid


class PR(SearchGrid):
    filename_suffix = "PR"
    xlabel = "Recall"
    ylabel = "Precision"
    colorbar_label = "max $F_1$"
    color_range = (28, 97)

    def summary_measure(self, sweep: ThresholdSweep):
        return max(sweep.F1)

    def plot_in_cell(self, ax: Axes, sweep: ThresholdSweep):
        ax.plot(
            self.sota_sweep.recall,
            self.sota_sweep.precision,
            c=self.sota_color,
            lw=1.5,
        )
        ax.plot(sweep.recall, sweep.precision, c=self.GEVec_color)
        ax.set_aspect("equal")
        ax.set_xlim(0.75, 1)
        ax.set_ylim(0.75, 1)
