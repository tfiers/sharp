from abc import ABC, abstractmethod
from colorsys import rgb_to_hls
from itertools import product
from logging import getLogger
from typing import Optional, Tuple, Sequence

from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from numpy import percentile, mean
from sharp.config.load import config
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.plot.results.base import (
    MultiEnvelopeFigureMaker,
)
from sharp.data.types.style import fraction
from sharp.tasks.plot.util.channelmap import draw_channelmap
from sharp.tasks.plot.util.legend import add_colored_legend
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF
from sharp.util.misc import cached

log = getLogger(__name__)


class SearchGrid(MultiEnvelopeFigureMaker, ABC):
    num_delays = (0, 1, 2, 3, 5, 10, 20, 40)
    channel_combo_names = tuple(config.channel_combinations.keys())
    _arg_tuples = tuple(product(num_delays, channel_combo_names))
    xlabel = "Included channels"
    ylabel = "Number of delays"

    filename_suffix: str
    colorbar_label: str
    legend_text: Optional[str] = None
    cell_xlabel: str
    cell_ylabel: str
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

    def work(self):
        self.plot_grid()
        self.plot_colorbar()

    def plot_grid(self):
        num_gridrows = len(self.num_delays)
        nrows = num_gridrows + 1
        ncols = len(self.channel_combo_names)
        figwidth = 1 + 1.8 * ncols
        figheight = 1 + self.rowheight * nrows
        rel_rowheight = SearchGrid.rowheight / self.rowheight
        channelmap_rel_height = 2.5 * rel_rowheight
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
        self.label_cell_x(axes[0, 0])
        self.label_cell_y(axes[0, -1])
        add_colored_legend(
            fig,
            ("GEVec", "Online BPF"),
            (self.GEVec_color, self.sota_color),
            ncol=2,
            loc="upper center",
        )
        self.label_grid(fig, axes)
        rect = (0.07, rel_rowheight * 0.013, 1, 1 - rel_rowheight * 0.04)
        fig.tight_layout(w_pad=self.col_pad, rect=rect)
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
            format=fraction,
        )
        fig.tight_layout()
        self.output_colorbar.write(fig)

    def label_cell_x(self, ax):
        ax.xaxis.set_major_formatter(fraction)
        ax.set_xticks(ax.get_xlim())
        ax.grid(False)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel(self.cell_xlabel)

    def label_cell_y(self, ax):
        ax.yaxis.set_major_formatter(fraction)
        ax.set_yticks(ax.get_ylim())
        ax.grid(False)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(self.cell_ylabel)

    def label_grid(self, fig: Figure, axes: Sequence[Axes]):
        x_center = mean([get_pos(axes[0, 0]).xmin, get_pos(axes[0, -1]).xmax])
        y_center = mean([get_pos(axes[0, 0]).ymax, get_pos(axes[-2, 0]).ymin])
        kwargs = dict(ha="center", va="center", fontsize=22)
        fig.text(x_center, 0.01, self.xlabel, **kwargs)
        fig.text(
            0.03,
            y_center,
            self.ylabel,
            **kwargs,
            rotation=90,
            rotation_mode="anchor",
        )

    @property
    @cached
    def norm(self):
        measures = [
            self.summary_measure(sweep) for sweep in self.threshold_sweeps[:-1]
        ]
        return Normalize(*percentile(measures, self.color_range))


def get_pos(ax: Axes) -> Bbox:
    return ax.get_position(True)
