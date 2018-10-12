

from fklab.plot.plots import plot_events, plot_segments
from fklab.segments import Segment
from sharp.data.types.aliases import Axes, EventList
from sharp.data.types.intersection import SegmentEventIntersection


def add_segments(ax: Axes, segments: Segment, alpha=0.3, **kwargs):
    """ Draw vertical bands on the plot, one for each segment. """
    visible_segs = segments.intersection(ax.get_xlim())
    plot_segments(visible_segs, axes=ax, fullheight=True, alpha=alpha, **kwargs)


def add_events(ax: Axes, events: EventList, lw=1, **kwargs):
    """ Draw vertical lines on the plot, one for each event. """
    view = Segment(ax.get_xlim())
    intersection = SegmentEventIntersection(view, events)
    visible_events = events[intersection.event_is_in_seg]
    plot_events(
        visible_events, axes=ax, fullheight=True, linewidth=lw, **kwargs
    )
