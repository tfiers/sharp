from typing import Sequence

from fklab.segments import Segment
from sharp.data.types.aliases import Axes
from sharp.data.types.intersection import SegmentEventIntersection
from sharp.tasks.plot.base import MultiEnvelopeFigureMaker
from sharp.tasks.plot.signals.base import TimeRange, TimeRangesPlotter
from sharp.tasks.plot.util.annotations import add_events


class PlotEnvelopes(MultiEnvelopeFigureMaker, TimeRangesPlotter):
    """
    Plots synched time slices of the input signal and the algorithm output
    envelopes.

    Annotates these time series plots with reference segments and detection
    times of all algorithms.
    """

    @property
    def output_dir(self):
        parent_dir = super(MultiEnvelopeFigureMaker, self).output_dir
        return parent_dir / "envelopes"

    @property
    def extra_signals(self):
        return self.evaluation.envelopes

    def include(self, time_range: TimeRange):
        return self.contains_detection(time_range)

    def contains_detection(self, time_range: TimeRange) -> bool:
        return any(
            SegmentEventIntersection(
                Segment(time_range), sweep.best.detections
            ).num_events_in_seg
            > 0
            for sweep in self.threshold_sweeps
        )

    def post_plot(
        self, time_range: TimeRange, input_ax: Axes, extra_axes: Sequence[Axes]
    ):
        for sweep, ax in zip(self.threshold_sweeps, extra_axes):
            te = sweep.best
            ax.hlines(
                sweep.best.threshold,
                *time_range,
                clip_on=False,
                linestyles="dashed"
            )
            add_events(ax, sweep.best.correct_detections, color="C2")
            add_events(ax, sweep.best.incorrect_detections, color="C1")
