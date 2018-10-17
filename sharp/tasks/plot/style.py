from matplotlib import style, cycler

# fmt: off
from seaborn import color_palette

readable = {
    'figure.figsize': (7, 5),
    'figure.dpi': 300,
    'savefig.dpi': 300,

    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,

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
}

symposium = {**readable, **griddy, **colourful}

style.use(symposium)
