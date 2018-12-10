from abc import abstractmethod
from typing import Sequence, List, Optional, Dict

from luigi import Parameter, BoolParameter
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from numpy import arange

from sharp.data.hardcoded.filters.base import num_delays_BPF, LTIRippleFilter
from sharp.data.types.aliases import subplots
from sharp.data.hardcoded.style import paperfig
from sharp.tasks.base import CustomParameter
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.plot.util.legend import add_colored_legend


class PlotSearchLines(FigureMaker):
    num_delays: Sequence[int]
    labels: List[str]
    legend_title: Optional[str] = CustomParameter(default=None)
    sequential_colors = BoolParameter(default=False)
    with_reference = BoolParameter(default=False)
    legend_outside = True
    linestyle = ".-"
    reference_color = "silver"

    @property
    def colors(self) -> List:
        num_colors = len(self.labels)
        if self.sequential_colors:
            cmap = get_cmap("viridis", num_colors)
            return list(cmap.colors)
        else:
            return [f"C{i}" for i in range(num_colors)]

    def work(self):
        fig, axes = subplots(nrows=2, sharex=True, figsize=paperfig())
        ax_top, ax_btm = axes
        ax_btm.set_xlabel("Number of delays")
        self.plot_on_axes(ax_top, ax_btm)
        self.add_legend(fig)
        fig.tight_layout()
        self.output().write(fig)

    def add_legend(self, fig):
        if self.legend_outside:
            loc_kwargs = dict(loc="center left", bbox_to_anchor=(0.98, 0.5))
        else:
            loc_kwargs = dict(loc="best")
        labels = self.labels
        colors = self.colors
        if self.with_reference:
            labels += ["Reference"]
            colors += [self.reference_color]
        add_colored_legend(
            parent=fig,
            labels=labels,
            colors=colors,
            title=self.legend_title,
            **loc_kwargs,
        )

    @abstractmethod
    def plot_on_axes(self, ax_top: Axes, ax_btm: Axes):
        ...


class BPF_SearchLines_Mixin:
    filename = Parameter()
    filters: Dict[str, LTIRippleFilter] = CustomParameter()
    orders = arange(14)
    num_delays = num_delays_BPF(orders)

    @property
    def labels(self):
        return list(self.filters.keys())
