from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from numpy import argmax
from seaborn import kdeplot
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.tasks.plot.results.searchgrid.base import SearchGrid
from sharp.util import cached


class Latency(SearchGrid):
    filename_suffix = "delay"
    xlabel = "Latency"
    ylabel = ""
    cmap = get_cmap("viridis_r")
    rowheight = 1.3
    text_kwargs = dict(x=0.99, y=0.94, ha="right", va="top")
    colorbar_label = "Median latency"
    legend_text = "Median"
    col_pad = 1.6
    color_range = (28, 95)

    def summary_measure(self, sweep: ThresholdSweep) -> float:
        return sweep.at_max_F1().rel_delays_median

    def plot_in_cell(self, ax: Axes, sweep: ThresholdSweep):
        kdeplot(self.sota_latencies, ax=ax, color=self.sota_color)
        kdeplot(sweep.at_max_F1().rel_delays, ax=ax, color=self.GEVec_color)
        ax.set_xlim(0, 1)

    @property
    @cached
    def sota_latencies(self):
        index_max_F1 = argmax(self.sota_sweep.F1)
        return self.sota_sweep.threshold_evaluations[index_max_F1].rel_delays

    def add_legend_hook(self, ax: Axes):
        ax.set_yticks([])
