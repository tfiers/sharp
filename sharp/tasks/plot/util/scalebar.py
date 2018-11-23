# coding: utf8


from matplotlib.axes import Axes
from matplotlib.text import Text
from numpy import diff, sign

from sharp.tasks.plot.util.sizing import points_to_figcoords


def add_time_scalebar(
    ax: Axes,
    length: float = 100,
    units: str = "ms",
    label: str = "{length:g} {units}",
    pos_along: float = 0.35,
    pos_across: float = 0.09,
    **kwargs,
):
    """
    Indicate time scale of an axes.

    :param ax: x-data should be in seconds.
    :param length: Length of the scalebar. In either milliseconds (default) or
                seconds, according to `units`.
    :param units: One of {'ms', 's'}.
    :param label: Format string for scalebar label text.
    :param kwargs: Passed on to `add_scalebar`.
    """
    if units == "s":
        scale = 1
    elif units == "ms":
        scale = 1000
    add_scalebar(
        ax,
        direction="h",
        length=length / scale,
        label=label.format(**locals()),
        pos_along=pos_along,
        pos_across=pos_across,
        **kwargs,
    )


def add_voltage_scalebar(
    ax: Axes,
    length: float = 500,
    units: str = "uV",
    label: str = "{length:g} {units}",
    pos_along: float = 0.3,
    pos_across: float = 0,
    **kwargs,
):
    """
    Indicate voltage scale of an axes.

    :param ax: y-data should be in microvolts.
    :param length: Size of the scalebar. In either microvolts (default) or
                millivolts, depending on `units`.
    :param units: One of {'uV', 'mV'}.
    :param label: Format string for scalebar label text. "uV" will be rendered
                as "μV".
    :param kwargs: Passed on to `add_scalebar`.
    """
    if units == "mV":
        scale = 1000
    elif units == "uV":
        scale = 1
    add_scalebar(
        ax,
        direction="v",
        length=length * scale,
        label=label.format(**locals()).replace("uV", "μV"),
        pos_along=pos_along,
        pos_across=pos_across,
        **kwargs,
    )


def add_scalebar(
    ax: Axes,
    direction: str,
    length: float,
    label: str,
    align: str = "left",
    pos_along: float = 0.1,
    pos_across: float = 0.12,
    label_size: float = 10,
    label_align: str = "center",
    label_pad: float = 0.2,
    label_offset: float = -0.75,
    lw: float = 1.4,
    brackets: bool = True,
    bracket_length: float = 3,
    label_background: dict = dict(facecolor="white", edgecolor="none", alpha=1),
):
    """
    Indicate horizontal or vertical scale of an axes.
    
    :param direction:  Orientation of the scalebar. "h" (for horizontal) to
                label the x-axis, "v" (for vertical) to label the y-axis.
    :param length:  Size of the scalebar, in data coordinates.
    :param label:  Text to label the scalebar with.
    :param align:  Alignment of the scalebar, in the direction of the scalebar.
    :param pos_along:  Position of the center of the scalebar, in the direction
                of the scalebar. In axes coordinates.
    :param pos_across:  Position of the center of the scalebar, in the
                direction orthogonal to the scalebar. In axes coordinates.
    :param label_size:  Fontsize (text height) of the label. In points (1/72-th
                of an inch).
    :param label_align:  {"left", "center", "right"}. Alignment of the label
                text along the writing direction, relative to the scalebar.
    :param label_pad:   When `label_align` is "left" or "right": padding of the
                label against the edge of the scalebar. In `label_size` units.
                Ignored when `label_align` is "center".
    :param label_offset:  Position of the center-line of the label text,
                relative to the scalebar. In `label_size` units. Negative
                values position the label to the left or below the scalebar.
    :param lw:  Line width of the scalebar.
    :param brackets:  Whether to end the scalebar in short perpendicular lines
                at both ends.
    :param bracket_length:  In points (1/72-th of an inch).
    :param label_background:  Passed on to `set_bbox()` of the label text.
    """
    if direction == "h":
        trans = ax.get_xaxis_transform()
        lims = ax.get_xlim()
        orthogonal_direction = "v"
    elif direction == "v":
        trans = ax.get_yaxis_transform()
        lims = ax.get_ylim()
        orthogonal_direction = "h"
    # `bar_start` and `bar_end` are in data coordinates.
    bar_start = lims[0] + (pos_along * diff(lims))
    bar_end = bar_start + length
    plot_options = dict(
        c="black",
        lw=lw,
        clip_on=False,
        transform=trans,
        zorder=4,
        solid_capstyle="projecting",
    )
    text_options = dict(
        va="center", ha=label_align, transform=trans, fontsize=label_size
    )
    fig = ax.get_figure()
    bracket_length_axcoords = points_to_figcoords(
        bracket_length, fig, trans=ax.transAxes, direction=orthogonal_direction
    )
    label_size_datacoords = points_to_figcoords(
        label_size, fig, trans=ax.transData, direction=direction
    )
    label_size_axcoords = points_to_figcoords(
        label_size, fig, trans=ax.transAxes, direction=orthogonal_direction
    )
    label_pos_across = pos_across + (label_offset * label_size_axcoords)
    if label_align == "left":
        label_pos_along = bar_start + (label_pad * label_size_datacoords)
    elif label_align == "center":
        label_pos_along = (bar_start + bar_end) / 2
    elif label_align == "right":
        label_pos_along = bar_end - (label_pad * label_size_datacoords)
    text_coords = (label_pos_along, label_pos_across)
    bar_coords = ([bar_start, bar_end], [pos_across, pos_across])
    # Format of below coords: imagine a horizontal bar, then: ([x, x], [y, y]).
    label_direction = sign(label_offset)
    bracket_direction = -label_direction
    bracket_start_coords = (
        [bar_start, bar_start],
        [pos_across, pos_across + bracket_direction * bracket_length_axcoords],
    )
    bracket_end_coords = (
        [bar_end, bar_end],
        [pos_across, pos_across + bracket_direction * bracket_length_axcoords],
    )
    if direction == "v":
        bar_coords = reversed(bar_coords)
        text_coords = reversed(text_coords)
        bracket_start_coords = reversed(bracket_start_coords)
        bracket_end_coords = reversed(bracket_end_coords)
        text_options.update(dict(rotation=90, rotation_mode="anchor"))
    ax.plot(*bar_coords, **plot_options)
    if brackets:
        ax.plot(*bracket_start_coords, **plot_options)
        ax.plot(*bracket_end_coords, **plot_options)
    text: Text = ax.text(*text_coords, label, **text_options)
    text.set_bbox(label_background)
