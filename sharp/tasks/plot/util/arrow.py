import numpy as np
from matplotlib.axes import Axes

import sharp.tasks.plot.util.sizing as util


def add_arrow(ax: Axes, x, y, rot=60, length=20, coords="axes"):
    """
    Draws an arrow on an axes.

    Parameters
    ----------
    ax : mpl.axes.Axes
    x, y : float
        Where the arrow points at (location of the arrow end).
    rot : float
        Rotation of the arrow, in degrees.
    length : float
        Length of the arrow, in pixels.
    coords : "axes", "data"  or  mpl.transforms.Transform
        Coordinate system of `x` and `y`.
    """
    if coords == "axes":
        trans = ax.transAxes
    elif coords == "data":
        trans = ax.transData
    rot_rad = rot * np.pi / 180
    length_coords = util.pixels_to_coords(length, trans)
    vector = length_coords * np.array([np.cos(rot_rad), np.sin(rot_rad)])
    end = (x, y)
    base = end - vector
    ax.annotate(
        "",
        end,
        base,
        xycoords=trans,
        textcoords=trans,
        arrowprops=dict(fc="black"),
    )
