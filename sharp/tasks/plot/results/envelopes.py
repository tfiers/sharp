from typing import Sequence

from matplotlib.axes import Axes
from numpy import ndarray

from fklab.segments import Segment
from sharp.data.types.intersection import SegmentEventIntersection
from sharp.tasks.plot.base import TimeRange, TimeRangesPlotter
from sharp.tasks.plot.results.base import MultiEnvelopeFigureMaker
from sharp.tasks.plot.util.annotations import add_event_arrows


class PlotEnvelopes(MultiEnvelopeFigureMaker, TimeRangesPlotter):
    """
    Plots synched time slices of the input signal and the algorithm output
    envelopes.

    Annotates these time series plots with reference segments and detection
    times of all algorithms.
    """

    def requires(self):
        r1 = MultiEnvelopeFigureMaker.requires(self)
        r2 = TimeRangesPlotter.requires(self)
        return tuple(r1) + tuple(r2)

    @property
    def output_dir(self):
        return super().output_dir / "envelopes"

    @property
    def extra_signals(self):
        return self.test_envelopes

    def is_included(self, time_range: TimeRange):
        return self._contains_any_detection(time_range)

    def _contains_any_detection(self, time_range: TimeRange) -> bool:
        seg = Segment(time_range)
        for sweep in self.threshold_sweeps:
            te = sweep.at_max_F1()
            if _contains_at_least_one(
                seg, te.correct_detections
            ) or _contains_at_least_one(seg, te.incorrect_detections):
                return True
        else:
            return False

    def post_plot(
        self, time_range: TimeRange, input_ax: Axes, extra_axes: Sequence[Axes]
    ):
        tups = zip(self.threshold_sweeps, extra_axes, self.titles, self.colors)
        for sweep, ax, title, color in tups:
            ax.hlines(
                sweep.at_max_F1().threshold,
                *time_range,
                clip_on=False,
                linestyles="dashed",
                linewidth=0.8,
            )
            add_event_arrows(
                ax, sweep.at_max_F1().correct_detections, color="green"
            )
            add_event_arrows(
                ax, sweep.at_max_F1().incorrect_detections, color="red"
            )
            ax.text(0.03, 0.74, title, color=color, transform=ax.transAxes)


def _contains_at_least_one(seg: Segment, events: ndarray):
    return SegmentEventIntersection(seg, events).num_events_in_seg > 0
