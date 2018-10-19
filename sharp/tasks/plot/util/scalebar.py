# coding: utf8

import numpy as np
from matplotlib.axes import Axes
from matplotlib.text import Text

from sharp.tasks.plot.util.sizing import get_fontsize, pixels_to_coords


def add_time_scalebar(
    ax: Axes,
    length: float = 100,
    units: str = "ms",
    pos_along: float = 0.35,
    pos_across: float = 0.09,
    **kwargs,
):
    """
    Indicate time scale of an axes.

    :param ax: x-data should be in seconds.
    :param length: Length of the scalebar. In either milliseconds (default) or
            seconds, according to `units`.
    :param units: {'ms', 's'}
    :param kwargs: Passed on to `add_scalebar`.
    """
    if units == "s":
        scale = 1
    elif units == "ms":
        scale = 1000
    add_scalebar(
        ax,
        orient="h",
        length=length / scale,
        label=f"{length:g} {units}",
        pos_along=pos_along,
        pos_across=pos_across,
        **kwargs,
    )


def add_voltage_scalebar(
    ax: Axes,
    length: float = 500,
    units: str = "uV",
    pos_along: float = 0.3,
    pos_across: float = 0,
    **kwargs,
):
    """
    Indicate voltage scale of an axes.

    :param ax: y-data should be in microvolts.
    :param length: Size of the scalebar. In either microvolts (default) or
        millivolts, depending on `units`.
    :param units: {'uV', 'mV'}
    :param kwargs: Passed on to `add_scalebar`.
    """
    if units == "mV":
        scale = 1000
    elif units == "uV":
        units = "μV"
        scale = 1
    add_scalebar(
        ax,
        orient="v",
        length=length * scale,
        label=f"{length:g} {units}",
        pos_along=pos_along,
        pos_across=pos_across,
        **kwargs,
    )


def add_scalebar(
    ax: Axes,
    orient: str,
    length: float,
    label: str,
    pos_along: float = 0.1,
    pos_across: float = 0.12,
    label_pad: float = 0.2,
    label_offset: float = -0.8,
    ha: str = "center",
    lw: float = 2,
    brackets: bool = True,
    bracket_length: float = 0.3,
    label_background: dict = dict(facecolor="white", edgecolor="none", alpha=1),
):
    """
    Indicate horizontal or vertical scale of an axes.

    :param orient: {'h', 'v'}. Direction of the scalebar.
    :param length: Size of the scalebar, in data coordinates.
    :param label: Text to label the scalebar with.
    :param pos_along: Position of the scalebar, in the direction of the scalebar.
        In axes coordinates.
    :param pos_across: Position of the scalebar, in the direction orthogonal to
        the scalebar. In axes coordinates.
    :param label_pad:  Padding of the label against the edge of the scalebar.
        In font size units (see notes). Ignored when `ha` is "center".
    :param label_offset: Relative position of the label orthogonal to the
        scalebar. In font size units (see notes).
    :param ha: {"left", "center", "right"}. Horizontal alignment of the label
        w.r.t. the scalebar, along the writing direction.
    :param lw: Line width of the scalebar.
    :param brackets: Whether to end the scalebar in short perpendicular lines
        at both ends.
    :param bracket_length: In font size units (see notes).
    :param label_background: Passed on to `set_bbox()` of the label text.

    Notes
    -----
    A parameter in "font size units" means that its value should be given as a
    fraction of Matplotlib's rcParams["font.size"].
    Scaling certain parameters in proportion to the font size guarantees that
    the scalebar looks good over a range of different font sizes and DPI
    settings.
    """
    if orient == "h":
        trans = ax.get_xaxis_transform()
        lims = ax.get_xlim()
        orient_across = "v"
    elif orient == "v":
        trans = ax.get_yaxis_transform()
        lims = ax.get_ylim()
        orient_across = "h"

    # `bar_start` and `bar_end` are in data coordinates.
    bar_start = lims[0] + (pos_along * np.diff(lims))
    bar_end = bar_start + length
    plot_options = dict(
        c="black", lw=lw, clip_on=False, transform=trans, zorder=4
    )
    text_options = dict(va="center", ha=ha, transform=trans)

    text_height = get_fontsize()
    text_height_datacoords = pixels_to_coords(text_height, ax.transData, orient)
    if ha == "left":
        label_pos_along = bar_start + (label_pad * text_height_datacoords)
    elif ha == "center":
        label_pos_along = (bar_start + bar_end) / 2
    elif ha == "right":
        label_pos_along = bar_end - (label_pad * text_height_datacoords)

    text_height_axcoords = pixels_to_coords(
        text_height, ax.transAxes, orient_across
    )
    label_pos_across = pos_across + (label_offset * text_height_axcoords)

    bracket_length_axcoords = pixels_to_coords(
        bracket_length * text_height, ax.transAxes, orient_across
    )

    bar_coords = ([bar_start, bar_end], [pos_across, pos_across])
    text_coords = (label_pos_along, label_pos_across)
    bracket_start_coords = (
        [bar_start, bar_start],
        [pos_across, pos_across + bracket_length_axcoords],
    )
    bracket_end_coords = (
        [bar_end, bar_end],
        [pos_across, pos_across + bracket_length_axcoords],
    )

    if orient == "v":
        text_options.update(dict(rotation=90, rotation_mode="anchor"))
        bar_coords = reversed(bar_coords)
        text_coords = reversed(text_coords)
        bracket_start_coords = reversed(bracket_start_coords)
        bracket_end_coords = reversed(bracket_end_coords)

    ax.plot(*bar_coords, **plot_options)
    if brackets:
        ax.plot(*bracket_start_coords, **plot_options)
        ax.plot(*bracket_end_coords, **plot_options)

    text: Text = ax.text(*text_coords, label, **text_options)
    text.set_bbox(label_background)
