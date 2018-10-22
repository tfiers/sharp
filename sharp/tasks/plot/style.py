from matplotlib import style, cycler
from seaborn import color_palette


# fmt: off
readable = {
    'figure.figsize': (7, 5),
    'figure.dpi': 80,
    'savefig.dpi': 300,

    'font.size': 15,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'legend.fontsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,

    'axes.titlepad': 30,
    'axes.labelpad': 22,

    'lines.linewidth': 3,
    'lines.solid_capstyle': 'round',
}

griddy = {
    'axes.grid': True,
    'grid.color': 'D0D0D0',

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
    'xtick.color': '909090',
    'ytick.color': '909090',
}

symposium = {**readable, **griddy, **colourful}
style.use(symposium)
