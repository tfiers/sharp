from math import tau
from typing import Optional, Tuple

from matplotlib.axes import Axes
from matplotlib.transforms import Transform
from numpy import array, cos, sin

from sharp.tasks.plot.util.sizing import points_to_figcoords


def add_arrow(
    ax: Axes,
    tip: Tuple[float, float],
    rot: float = 60,
    length: float = 20,
    trans: Optional[Transform] = None,
):
    """
    Highlight part of a plot by pointing an arrow to it.

    :param ax:  Axes to plot on.
    :param tip:  Where the arrow points at (location of the arrow end).
    :param rot:  Rotation of the arrow, in degrees. 0 is pointing east.
    :param length:  Lenght of the arrow, in points (1/72-th of an inch).
    :param trans:  Coordinate system of `point`. Default: `ax.transAxes`, for
                axes coordinates. Other interesting options: `ax.transData`,
                `ax.get_xaxis_transform()`, and `ax.get_yaxis_transform()`.
    :return:
    """
    if trans is None:
        trans = ax.transAxes
    tip = array(tip)
    angle = (rot / 360) * tau
    vector_length = points_to_figcoords(length, trans)
    vector = vector_length * array([cos(angle), sin(angle)])
    base = tip - vector
    ax.annotate(
        "",
        tip,
        base,
        xycoords=trans,
        textcoords=trans,
        arrowprops=dict(fc="black"),
    )
