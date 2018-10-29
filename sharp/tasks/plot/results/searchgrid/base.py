from abc import ABC, abstractmethod
from colorsys import rgb_to_hls
from itertools import product
from logging import getLogger
from typing import Optional, Tuple

from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.text import Text
from numpy import percentile, array
from sharp.config.load import config
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.plot.results.base import (
    MultiEnvelopeFigureMaker,
    fraction_formatter,
)
from sharp.tasks.plot.util.channelmap import draw_channelmap
from sharp.tasks.plot.util.legend import add_colored_legend
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF
from sharp.util import cached

log = getLogger(__name__)


class SearchGrid(MultiEnvelopeFigureMaker, ABC):
    num_delays = (0, 1, 2, 3, 5, 10, 20, 40)
    channel_combo_names = tuple(config.channel_combinations.keys())
    _arg_tuples = tuple(product(num_delays, channel_combo_names))

    filename_suffix: str
    colorbar_label: str
    legend_text: Optional[str] = None
    xlabel: str
    ylabel: str
    cmap = get_cmap("viridis")
    rowheight = 1.7
    text_kwargs = dict(x=0.04, y=0.04)
    text_format = "{summary_measure:.0%}"
    col_pad = 1.08
    color_range: Tuple[float, float] = (28, 97)
    # As percentiles of `summary_measure` for all convolver sweeps.
    GEVec_color = "black"
    sota_color = "grey"

    @property
    def convolvers(self):
        return tuple(
            SpatiotemporalConvolution(num_delays=num, channel_combo_name=name)
            for num, name in self._arg_tuples
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
            self.output_dir, f"{self.filename_suffix} -- searchgrid"
        )

    @property
    def output_colorbar(self):
        return FigureTarget(
            self.output_dir, f"{self.filename_suffix} -- colorbar"
        )

    @property
    def output_legend(self):
        return FigureTarget(
            self.output_dir, f"{self.filename_suffix} -- legend"
        )

    def output(self):
        return (self.output_grid, self.output_colorbar)
        # return (self.output_grid, self.output_colorbar, self.output_legend)

    def run(self):
        self.plot_grid()
        self.plot_colorbar()
        # self.plot_legend()

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
        # self.add_legend(axes)
        fig.tight_layout(w_pad=self.col_pad)
        self.output_grid.write(fig)

    def plot_grid_cells(self, axes):
        for row, num in enumerate(self.num_delays):
            for col, name in enumerate(self.channel_combo_names):
                index = self._arg_tuples.index((num, name))
                cell_name = self.convolvers[index].filename
                log.info(
                    f"Searchgrid '{self.filename_suffix}', ax: {cell_name}"
                )
                sweep = self.threshold_sweeps[index]
                ax: Axes = axes[row, col]
                self.plot_in_cell(ax, sweep)
                self.plot_cell_summary(ax, self.summary_measure(sweep))
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(num, rotation=0, labelpad=40)

    @abstractmethod
    def plot_in_cell(self, ax: Axes, sweep: ThresholdSweep):
        ...

    @abstractmethod
    def summary_measure(self, sweep: ThresholdSweep) -> float:
        ...

    def plot_cell_summary(self, ax: Axes, summary_measure: float) -> Text:
        # transform value in [vmin, vmax] to [0, 1]
        fraction = self.norm(summary_measure)
        # convert value in [0, 1] to an RGBA value
        facecolor = self.cmap(fraction)
        ax.set_facecolor(facecolor)
        _, lightness, _ = rgb_to_hls(*facecolor[:3])
        if lightness > 0.4:
            text_color = "black"
        else:
            text_color = "white"
        return ax.text(
            s=self.text_format.format(summary_measure=summary_measure),
            transform=ax.transAxes,
            color=text_color,
            **self.text_kwargs,
        )

    def plot_channelmaps(self, axes):
        for col, name in enumerate(self.channel_combo_names):
            ax = axes[-1, col]
            draw_channelmap(
                ax, active_channels=config.channel_combinations[name]
            )

    def plot_colorbar(self):
        fig, ax = subplots(figsize=(2, 4))
        cbar = ColorbarBase(
            ax=ax,
            label=self.colorbar_label,
            norm=self.norm,
            cmap=self.cmap,
            extend="both",
            format=fraction_formatter,
        )
        fig.tight_layout()
        self.output_colorbar.write(fig)

    # def add_legend(self, axes):
    #     ax: Axes = axes[0, -1]
    #     ax

    def plot_legend(self):
        figsize = 1.13 * array([2.8, 1.4 * self.rowheight])
        fig, ax = subplots(figsize=figsize)
        self.plot_in_cell(ax, sweep=None)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.xaxis.set_major_formatter(fraction_formatter)
        ax.yaxis.set_major_formatter(fraction_formatter)
        ax.set_xticks(ax.get_xlim())
        ax.set_yticks(ax.get_ylim())
        ax.grid(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        t = self.plot_cell_summary(ax, self.summary_measure(self.sota_sweep))
        if self.legend_text is None:
            self.legend_text = self.colorbar_label
        t.set_text(self.legend_text)
        self.plot_legend_hook(fig, ax)
        add_colored_legend(
            fig, ("Online BPF", "GEVec"), (self.sota_color, self.GEVec_color)
        )
        fig.tight_layout()
        self.output_legend.write(fig)

    def plot_legend_hook(self, fig: Figure, ax: Axes):
        pass

    @property
    @cached
    def norm(self):
        measures = [
            self.summary_measure(sweep) for sweep in self.threshold_sweeps[:-1]
        ]
        return Normalize(*percentile(measures, self.color_range))
