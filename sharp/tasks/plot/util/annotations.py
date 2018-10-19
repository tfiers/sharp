from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy import ones_like

from fklab.plot.plots import plot_events, plot_segments
from fklab.segments import Segment
from sharp.data.types.aliases import EventList
from sharp.data.types.intersection import SegmentEventIntersection


def add_segments(ax: Axes, segments: Segment, alpha=0.3, **kwargs):
    """ Draw vertical bands on the plot, one for each segment. """
    visible_segs = segments.intersection(ax.get_xlim())
    plot_segments(visible_segs, axes=ax, fullheight=True, alpha=alpha, **kwargs)


def add_event_lines(ax: Axes, events: EventList, lw=1, **kwargs):
    """ Draw vertical lines on the plot, one for each (visible) event. """
    plot_events(
        _get_visible_events(ax, events),
        axes=ax,
        fullheight=True,
        linewidth=lw,
        **kwargs
    )


def add_event_arrows(ax: Axes, events: EventList, **kwargs):
    """ Draw arrows beneath the plot, one for each (visible) event. """
    x = _get_visible_events(ax, events)
    y = -0.1 * ones_like(x)
    trans = ax.get_xaxis_transform()
    arrows: Sequence[Line2D] = ax.plot(
        x, y, "^", transform=trans, clip_on=False, **kwargs
    )
    # fig.tight_layout doesn't work when clip_on=False here. So we instruct
    # tight_layout to ignore the created artists
    for arrow in arrows:
        arrow.set_in_layout(False)


def _get_visible_events(ax, events):
    view = Segment(ax.get_xlim())
    intersection = SegmentEventIntersection(view, events)
    return events[intersection.event_is_in_seg]
