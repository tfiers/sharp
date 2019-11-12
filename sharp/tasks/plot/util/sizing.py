from typing import Tuple, Union

from matplotlib.figure import Figure
from matplotlib.transforms import Transform

INCHES_PER_POINT = 1 / 72


def points_to_pixels(length_points: float, fig: Figure) -> float:
    """
    Convert a length in points to a length in display coordinates.
    
    :param length_points:  An absolute size, in points (1/72-th of an inch).
    :param fig:  Provides the DPI (pixels-per-inch) setting for conversion.
    :return:  A length in display coordinates (i.e. pixels).
    """
    pixels_per_inch = fig.get_dpi()
    length_inches = length_points * INCHES_PER_POINT
    length_pixels = length_inches * pixels_per_inch
    return length_pixels


def pixels_to_figcoords(
    length_pixels: float, trans: Transform, direction: str = "both"
) -> Union[float, Tuple[float, float]]:
    """
    Convert a length in display coordinates to a length in data or axes
    coordinates.
    
    :param length_pixels:  A absolute size, in display coordinates (i.e.
                pixels).
    :param trans:  Coordinate system to transform to. Examples: `ax.transAxes`,
                `ax.transData`, `ax.get_xaxis_transform()`, and
                `ax.get_yaxis_transform()`.
    :param direction:  Which dimension to return. 'h' for horizontal (x), 'v' for
                vertical (y), or 'both' for both.
    :return:  Length in `trans` coordinates. A tuple when orient="both". A
                scalar otherwise.
    """
    # Get the origin in display coordinates (i.e. pixels):
    origin = trans.transform([0, 0])
    # Transform display coordinates to data or axes coordinates:
    x, y = trans.inverted().transform(origin + length_pixels)
    if direction == "h":
        return x
    elif direction == "v":
        return y
    elif direction == "both":
        return x, y


def points_to_figcoords(
    length_points: float, fig: Figure, trans: Transform, direction: str = "both"
) -> Union[float, Tuple[float, float]]:
    """
    Convert a length in points to a length in data or axes coordinates.
    
    :param length_points:  An absolute size, in points (1/72-th of an inch).
    :param fig:  Figure that is the source of `trans`, and in which the
                transformed length will be used.
    :param trans:  See :func:`~pixels_to_figcoords`.
    :param direction:  See :func:`~pixels_to_figcoords`.
    :return:  See :func:`~pixels_to_figcoords`.
    """
    length_pixels = points_to_pixels(length_points, fig)
    return pixels_to_figcoords(length_pixels, trans, direction)
