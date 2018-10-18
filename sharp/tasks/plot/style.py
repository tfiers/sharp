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
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,

    'axes.titlepad': 30,
    'axes.labelpad': 18,
}

griddy = {
    'axes.grid': True,
    'grid.color': 'C0C0C0',

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
    'axes.labelcolor': '606060',
    'xtick.color': 'A0A0A0',
    'ytick.color': 'A0A0A0',
}

symposium = {**readable, **griddy, **colourful}
style.use(symposium)
