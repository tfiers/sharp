from abc import ABC, abstractmethod
from colorsys import rgb_to_hls
from itertools import product
from logging import getLogger
from typing import Tuple

from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
from numpy import argmax, max, percentile

from seaborn import kdeplot
from sharp.config.load import config
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.tasks.base import SharpTask
from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.plot.results.base import MultiEnvelopeFigureMaker
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF
from sharp.util import cached

log = getLogger(__name__)


class PlotSearchGrids(SharpTask):
    def requires(self):
        kwargs = dict(subdir="space-time-comp")
        return (PRGrid(**kwargs), DelayGrid(**kwargs))
        # return DelayGrid(**kwargs)


class SearchGrid(MultiEnvelopeFigureMaker, ABC):
    num_delays = (0, 1, 2, 3, 5, 10, 20, 40)
    channel_combo_names = tuple(config.channel_combinations.keys())
    arg_tuples = tuple(product(num_delays, channel_combo_names))
    # __call__ converts a value in [0, 1] to an RGBA value.
    filename_suffix: str = ...
    cmap = get_cmap("viridis")
    rowheight = 1.7
    text_kwargs = dict(x=0.04, y=0.04)
    text_format = "{summary_measure:.3f}"
    legend_kwargs = dict()

    @property
    def convolvers(self) -> Tuple[SpatiotemporalConvolution, ...]:
        return tuple(
            SpatiotemporalConvolution(num_delays=num, channel_combo_name=name)
            for num, name in self.arg_tuples
        )

    sota = ApplyOnlineBPF()

    @property
    def envelope_makers(self):
        return self.convolvers + (self.sota,)

    @property
    def sota_sweep(self):
        return self.threshold_sweeps[-1]

    @property
    def output_grid(self):
        return FigureTarget(
            self.output_dir, f"searchgrid {self.filename_suffix}"
        )

    @property
    def output_legend(self):
        return FigureTarget(self.output_dir, f"legend {self.filename_suffix}")

    def output(self):
        return (self.output_grid, self.output_legend)

    def run(self):
        self.plot_grid()
        self.plot_legend()

    def plot_grid(self):
        num_gridrows = len(self.num_delays)
        nrows = num_gridrows + 1
        ncols = len(self.channel_combo_names)
        figwidth = 1 + 1.8 * ncols
        figheight = self.rowheight * nrows
        channelmap_rel_height = 2.5 * SearchGrid.rowheight / self.rowheight
        fig, axes = subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figwidth, figheight),
            gridspec_kw=dict(
                height_ratios=[1] * num_gridrows + [channelmap_rel_height]
            ),
        )
        self.plot_grid_cells(axes)
        self.plot_channelmaps(axes)
        fig.tight_layout(w_pad=1.2)
        self.output_grid.write(fig)

    def plot_grid_cells(self, axes):
        for row, num in enumerate(self.num_delays):
            for col, name in enumerate(self.channel_combo_names):
                index = self.arg_tuples.index((num, name))
                cell_name = self.convolvers[index].filename
                log.info(
                    f"Searchgrid '{self.filename_suffix}', ax: {cell_name}"
                )
                sweep = self.threshold_sweeps[index]
                ax: Axes = axes[row, col]
                self.plot_in_cell(sweep, ax)
                self.plot_cell_summary(ax, self.summary_measure(sweep))
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(num, rotation=0, labelpad=40)

    @abstractmethod
    def plot_in_cell(self, sweep: ThresholdSweep, ax: Axes):
        ...

    @abstractmethod
    def summary_measure(self, sweep: ThresholdSweep) -> float:
        ...

    def plot_cell_summary(self, ax: Axes, summary_measure: float):
        fraction = self.norm(summary_measure)
        facecolor = self.cmap(fraction)
        ax.set_facecolor(facecolor)
        _, lightness, _ = rgb_to_hls(*facecolor[:3])
        if lightness > 0.35:
            text_color = "black"
        else:
            text_color = "white"
        ax.text(
            s=self.text_format.format(summary_measure=summary_measure),
            transform=ax.transAxes,
            color=text_color,
            **self.text_kwargs,
        )

    def plot_channelmaps(self, axes):
        for col, name in enumerate(self.channel_combo_names):
            ax = axes[-1, col]
            config.draw_channelmap(
                ax, active_channels=config.channel_combinations[name]
            )

    def plot_legend(self):
        fig, ax = subplots(figsize=(2, 4))
        cbar = ColorbarBase(
            ax=ax, norm=self.norm, cmap=self.cmap, **self.legend_kwargs
        )
        self.plot_legend_hook(ax, cbar)
        fig.tight_layout()
        self.output_legend.write(fig)

    def plot_legend_hook(self, ax: Axes, cbar: ColorbarBase):
        pass

    @property
    @cached
    def norm(self):
        # __call__ transforms a value in [vmin, vmax] to [0, 1]
        return Normalize(*self.legend_lims)

    @property
    @abstractmethod
    def legend_lims(self) -> Tuple[float, float]:
        ...


class PRGrid(SearchGrid):
    filename_suffix = "PR"
    legend_kwargs = dict(label="max $F_1$", extend="min", extendfrac=0.1)

    def summary_measure(self, sweep: ThresholdSweep):
        return max(sweep.F1)

    def plot_in_cell(self, sweep: ThresholdSweep, ax: Axes):
        ax.plot(
            self.sota_sweep.recall, self.sota_sweep.precision, c="grey", lw=1.5
        )
        ax.plot(sweep.recall, sweep.precision, c="black")
        ax.set_aspect("equal")
        ax.set_xlim(0.75, 1)
        ax.set_ylim(0.75, 1)

    @property
    def legend_lims(self):
        max_F1s = [max(sweep.F1) for sweep in self.threshold_sweeps]
        # vmin = 0.8
        vmin = percentile(max_F1s, 28)
        vmax = percentile(max_F1s, 97)
        return (vmin, vmax)


class DelayGrid(SearchGrid):
    filename_suffix = "delay"
    cmap = get_cmap("viridis_r")
    rowheight = 1.3
    text_kwargs = dict(x=0.99, y=0.94, ha="right", va="top")
    text_format = "{summary_measure:.1%}"
    legend_kwargs = dict(
        label="Median latency",
        format=PercentFormatter(xmax=1, decimals=0),
        extend="max",
        extendfrac=0.1,
    )

    def summary_measure(self, sweep: ThresholdSweep) -> float:
        return sweep.at_max_F1().rel_delays_median

    def plot_in_cell(self, sweep: ThresholdSweep, ax: Axes):
        kdeplot(self.sota_delays, ax=ax, color="grey")
        kdeplot(sweep.at_max_F1().rel_delays, ax=ax, color="black")
        ax.set_xlim(0, 1)

    @property
    @cached
    def sota_delays(self):
        index_max_F1 = argmax(self.sota_sweep.F1)
        return self.sota_sweep.threshold_evaluations[index_max_F1].rel_delays

    @property
    def legend_lims(self):
        median_latencies = [
            sweep.at_max_F1().rel_delays_median
            for sweep in self.threshold_sweeps[:-1]
        ]
        vmin = percentile(median_latencies, 28)
        vmax = percentile(median_latencies, 97)
        return (vmin, vmax)

    def plot_legend_hook(self, ax: Axes, cbar: ColorbarBase):
        pass
        # ax.invert_yaxis()
