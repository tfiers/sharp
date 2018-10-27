from abc import ABC, abstractmethod
from itertools import product
from logging import getLogger
from typing import Tuple

from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
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


class SearchGrid(MultiEnvelopeFigureMaker, ABC):
    num_delays = (0, 1, 2, 3, 5, 10, 20, 40)
    channel_combo_names = tuple(config.channel_combinations.keys())

    arg_tuples = tuple(product(num_delays, channel_combo_names))

    # __call__ converts a value in [0, 1] to an RGBA value.
    cmap = get_cmap("viridis")

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

    filename_suffix: str = ...

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
        nrows = len(self.num_delays)
        ncols = len(self.channel_combo_names)
        figwidth = 1 + 1.8 * ncols
        figheight = 1.7 * nrows
        fig, axes = subplots(
            nrows=nrows, ncols=ncols, figsize=(figwidth, figheight)
        )
        last_row = nrows - 1
        for row, num in enumerate(self.num_delays):
            for col, name in enumerate(self.channel_combo_names):
                index = self.arg_tuples.index((num, name))
                cell_name = self.convolvers[index].filename
                log.info(f"{self.filename_suffix}, ax: {cell_name}")
                sweep = self.threshold_sweeps[index]
                ax: Axes = axes[row, col]
                self.plot_in_cell(sweep, ax)
                ax.set_xticks([])
                ax.set_yticks([])
                pad = 40
                if col == 0:
                    ax.set_ylabel(num, rotation=0, labelpad=pad)
                if row == last_row:
                    ax.set_xlabel(name, labelpad=pad)
        fig.tight_layout()
        self.output_grid.write(fig)

    @abstractmethod
    def plot_in_cell(self, sweep: ThresholdSweep, ax: Axes):
        ...

    def plot_legend(self):
        fig, ax = subplots(figsize=(2, 4))
        cbar = ColorbarBase(
            ax=ax,
            norm=self.F1_normalizer,
            cmap=self.cmap,
            extend="min",
            extendfrac=0.1,
        )
        cbar.set_label("max $F_1$")
        fig.tight_layout()
        self.output_legend.write(fig)

    @property
    @cached
    def F1_normalizer(self):
        # __call__ transforms a value in [vmin, vmax] to [0, 1]
        max_F1s = [max(sweep.F1) for sweep in self.threshold_sweeps]
        # vmin = 0.8
        vmin = percentile(max_F1s, 28)
        vmax = percentile(max_F1s, 97)
        return Normalize(vmin, vmax)


class DelayGrid(SearchGrid):
    filename_suffix = "delay"

    def plot_in_cell(self, sweep: ThresholdSweep, ax: Axes):
        max_F1 = max(sweep.F1)
        fraction = self.F1_normalizer(max_F1)
        ax.set_facecolor(self.cmap(fraction))
        ax.plot(
            self.sota_sweep.recall, self.sota_sweep.precision, c="grey", lw=1.5
        )
        ax.plot(sweep.recall, sweep.precision, c="black")
        if fraction > 0.5:
            text_color = "black"
        else:
            text_color = "white"
        ax.text(
            0.04,
            0.04,
            f"{max_F1:.3f}",
            transform=ax.transAxes,
            color=text_color,
        )
        ax.set_aspect("equal")
        ax.set_xlim(0.75, 1)
        ax.set_ylim(0.75, 1)


class PRGrid(SearchGrid):
    filename_suffix = "PR"

    def plot_in_cell(self, sweep: ThresholdSweep, ax: Axes):
        index_max_F1 = argmax(sweep.F1)
        delays = sweep.threshold_evaluations[index_max_F1].rel_delays
        kdeplot(self.sota_delays, ax=ax, color="grey")
        kdeplot(delays, ax=ax, color="black")
        ax.set_xlim(0, 1)

    @property
    @cached
    def sota_delays(self):
        index_max_F1 = argmax(self.sota_sweep.F1)
        return self.sota_sweep.threshold_evaluations[index_max_F1].rel_delays
