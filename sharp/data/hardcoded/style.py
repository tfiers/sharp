from typing import Tuple

from matplotlib import style, cycler
from matplotlib.ticker import PercentFormatter

from seaborn import color_palette
from numpy import array

# A4 dimensions, in inches
paper_width = 8.27
paper_height = 11.69
line_width = 0.7 * paper_width


def paperfig(width=1.0, height=0.7) -> Tuple[float, float]:
    """
    :return:  A figure size tuple, in inches.
    :param width:  Relative to paper linewidth.
    :param height:  Also relative to paper linewidth.
    """
    return line_width * array((width, height))


# Font sizes and linewidths are given in points, which are 1/72-th of an inch.

# fmt: off
readable = {
    'figure.figsize': paperfig(),
    
    # DPI (pixels-per-inch) is irrelevant for PDF's. But it does matter for
    # Jupyter Notebooks ('figure.dpi') and saved PNG's ('savefig.dpi').
    'figure.dpi': 80,
    'savefig.dpi': 300,

    'font.size': 8.5,
    'axes.labelsize': 10.5,
    'axes.titlesize': 10.5,
    'legend.fontsize': 11.5,
    'legend.title_fontsize': 11.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,

    'axes.titlepad': 10,
    'axes.labelpad': 9,

    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'lines.solid_capstyle': 'round',
}

griddy = {
    'axes.grid': True,
    'grid.color': 'D0D0D0',
    'grid.linewidth': 0.4,

    'xtick.bottom': False,
    'ytick.left': False,

    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.bottom': False,
    'axes.spines.left': False,
}

seaborn_colours = color_palette('muted')
blue, orange, green, red, purple, brown, pink, *others = seaborn_colours

colourful = {
    'axes.prop_cycle': cycler('color', seaborn_colours),
    'axes.labelcolor': '202020',
    'xtick.color': '707070',
    'ytick.color': '707070',
}

fraction = PercentFormatter(xmax=1, decimals=0)

symposium = {**readable, **griddy, **colourful}
style.use(symposium)
