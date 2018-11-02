from itertools import chain
from typing import Sequence, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from sharp.data.types.aliases import Color


def add_colored_legend(
    parent: Union[Figure, Axes],
    labels: Sequence[str] = None,
    colors: Sequence[Color] = None,
    **legend_kwargs,
):
    """
    Add a figure legend with colored labels, and without example artists.
    """
    legend: Legend = parent.legend(
        labels=labels,
        handlelength=0,
        handleheight=0,
        handletextpad=0,
        markerscale=0,
        labelspacing=0.7,
        borderpad=0.7,
        **legend_kwargs,
    )
    for artist in chain(legend.get_lines(), legend.get_patches()):
        # (Removing legend artists is not implemented in mpl).
        artist.set_visible(False)
    texts = legend.get_texts()
    if colors is None:
        colors = [f"C{i}" for i in range(len(texts))]
    for text, color in zip(texts, colors):
        text.set_color(color)
