from colorsys import rgb_to_hls
from logging import getLogger

from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.ticker import StrMethodFormatter
from numpy import arange, max, mean, percentile

from sharp.config.load import config
from sharp.data.files.figure import FigureTarget
from sharp.data.hardcoded.style import paperfig
from sharp.data.types.aliases import subplots
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.tasks.base import TaskParameter
from sharp.tasks.evaluate.sweep import ThresholdSweeper
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.plot.paper import output_dir
from sharp.tasks.signal.base import EnvelopeMaker
from sharp.tasks.signal.reference import MakeReference
from sharp.util.misc import cached

log = getLogger(__name__)


class PaperGridPlotter(FigureMaker):
    envelope_maker: EnvelopeMaker = TaskParameter()
    reference_makers = [
        MakeReference(**args) for args in MakeReference.args
    ]
    cmap = get_cmap("viridis")
    colorbar_label: str = ...
    fstring: str = ...

    @property
    def sweepers(self):
        return [
            ThresholdSweeper(
                envelope_maker=self.envelope_maker, reference_maker=rm
            )
            for rm in self.reference_makers
        ]

    def requires(self):
        return self.sweepers

    def output(self):
        return (self.output_grid, self.output_colorbar)

    @property
    def output_grid(self):
        return FigureTarget(output_dir, self.filename)

    @property
    def output_colorbar(self):
        return FigureTarget(output_dir / "cbar", self.filename)

    @property
    def filename(self):
        return f"{self.__class__.__name__} -- {self.envelope_maker.title}"

    def work(self):
        log.info(
            (
                self.filename,
                mean(self.data_matrix),
                percentile(self.data_matrix, [25, 50, 100]),
            )
        )
        self.plot_grid()
        self.plot_colorbar()

    def plot_colorbar(self):
        fig, ax = subplots(figsize=paperfig(0.42, 0.16))
        cbar = ColorbarBase(
            ax=ax,
            orientation="horizontal",
            label=self.colorbar_label,
            norm=self.norm,
            cmap=self.cmap,
            extend="both",
            format=StrMethodFormatter(self.fstring),
        )
        fig.tight_layout()
        self.output_colorbar.write(fig)

    def plot_grid(self):
        fig, ax = subplots(figsize=paperfig(0.55, 0.55))
        ax.imshow(
            self.data_matrix,
            cmap=self.cmap,
            origin="lower",
            aspect="auto",
            norm=self.norm,
        )
        ax.set_xlabel("Ripple definition (μV)")
        ax.set_ylabel("Sharp wave definition (μV)")
        num_ripple = len(self.ripple_thresholds)
        num_SW = len(self.SW_thresholds)
        ax.set_xticks(arange(num_ripple)[::2])
        ax.set_yticks(arange(num_SW)[::2])
        ax.set_xticklabels([f"{x:.0f}" for x in self.ripple_thresholds][::2])
        ax.set_yticklabels([f"{x:.0f}" for x in self.SW_thresholds][::2])
        for SW_ix in range(num_SW):
            for ripple_ix in range(num_ripple):
                value = self.data_matrix[SW_ix][ripple_ix]
                text = self.fstring.format(x=value)
                ax.text(
                    ripple_ix,
                    SW_ix,
                    text,
                    ha="center",
                    va="center",
                    color=self.text_color(value),
                    size="smaller",
                )
        ax.grid(False)
        fig.tight_layout()
        self.output_grid.write(fig)

    @property
    @cached
    def SW_thresholds(self):
        return [
            MakeReference(
                mult_detect_SW=mult_SW,
                mult_detect_ripple=config.mult_detect_ripple[0],
            ).threshold_detect_SW
            for mult_SW in config.mult_detect_SW
        ]

    @property
    @cached
    def ripple_thresholds(self):
        return [
            MakeReference(
                mult_detect_SW=config.mult_detect_SW[0],
                mult_detect_ripple=mult_ripple,
            ).threshold_detect_ripple
            for mult_ripple in config.mult_detect_ripple
        ]

    @property
    @cached
    def data_matrix(self):
        return [
            [
                self.get_data(
                    ThresholdSweeper(
                        envelope_maker=self.envelope_maker,
                        reference_maker=MakeReference(
                            mult_detect_ripple=ripple, mult_detect_SW=SW
                        ),
                    )
                    .output()
                    .read()
                )
                for ripple in config.mult_detect_ripple
            ]
            for SW in config.mult_detect_SW
        ]

    def get_data(self, sweep: ThresholdSweep) -> float:
        ...

    @property
    @cached
    def norm(self):
        return Normalize(*self.cmap_range)

    def text_color(self, data_value):
        # transform data_value in [vmin, vmax] to [0, 1]
        fraction = self.norm(data_value)
        # convert value in [0, 1] to an RGBA value
        facecolor = self.cmap(fraction)
        _, lightness, _ = rgb_to_hls(*facecolor[:3])
        if lightness > 0.4:
            text_color = "0.2"
        else:
            text_color = "0.8"
        return text_color


class AccuracyGrid(PaperGridPlotter):
    cmap_range = (0.62, 0.98)
    cmap = get_cmap("plasma")
    colorbar_label = "max $F_2$"
    fstring = "{x:.0%}"

    def get_data(self, sweep):
        return max(sweep.F2)


class LatencyGrid(PaperGridPlotter):
    cmap_range = (8, 32)
    cmap = get_cmap("viridis_r")
    colorbar_label = "Detection latency (ms)"
    fstring = "{x:.0f}"

    def get_data(self, sweep):
        return sweep.at_max_F2().abs_delays_median * 1000


class RelativeLatencyGrid(PaperGridPlotter):
    cmap_range = (0.15, 0.48)
    cmap = get_cmap("viridis_r")
    colorbar_label = "Detection latency"
    fstring = "{x:.0%}"

    def get_data(self, sweep):
        return sweep.at_max_F2().rel_delays_median
