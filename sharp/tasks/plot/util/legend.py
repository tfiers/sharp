from typing import Sequence, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def add_colored_legend(
    artist: Union[Figure, Axes],
    labels: Sequence[str] = None,
    colors: Sequence[str] = None,
    **legend_kwargs,
):
    """
    Add a figure legend with colored labels, and without example artists.
    """
    legend = artist.legend(
        labels=labels,
        handlelength=0,
        handletextpad=0,
        markerscale=0,
        labelspacing=0.9,
        borderpad=0.9,
        **legend_kwargs,
    )
    texts = legend.get_texts()
    if colors is None:
        colors = [f"C{i}" for i in range(len(texts))]
    for text, color in zip(texts, colors):
        text.set_color(color)
