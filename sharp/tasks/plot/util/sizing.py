from typing import Tuple, Union

from matplotlib.figure import Figure
from matplotlib.transforms import Transform


def transform_inches(
    length: float, fig: Figure, trans: Transform, orient: str = "both"
) -> Union[float, Tuple[float, float]]:
    """
    Convert a length in inches to a length in data or axes coordinates.
    
    :param length:  A size, in inches.
    :param fig:  Figure that is the source of `trans`, and in which the
                transformed length will be used.
    :param trans:  Coordinate system to transform to. Examples: `ax.transAxes`,
                `ax.transData`, `ax.get_xaxis_transform()`, and
                `ax.get_yaxis_transform()`.
    :param orient:  Which dimension to return. 'h' for horizontal, 'v' for
                vertical, or 'both' for both.
    :return:  Length in `trans` coordinates. A tuple when orient="both". A
                scalar otherwise.
    """
    pixels_per_inch = fig.get_dpi()
    length_pixels = length * pixels_per_inch
    # Get the origin in display coordinates (pixels):
    origin = trans.transform([0, 0])
    # Transform display coordinates to data or axes coordinates:
    x, y = trans.inverted().transform(origin + length_pixels)
    if orient == "h":
        return x
    elif orient == "v":
        return y
    elif orient == "both":
        return x, y
