from typing import Sequence

from matplotlib.axes import Axes

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
        d1 = super(MultiEnvelopeFigureMaker, self).requires()
        d2 = super(TimeRangesPlotter, self).requires()
        return tuple(d1) + tuple(d2)

    @property
    def output_dir(self):
        return super().output_dir / "envelopes"

    @property
    def extra_signals(self):
        return self.test_envelopes

    def include(self, time_range: TimeRange):
        return self.contains_detection(time_range)

    def contains_detection(self, time_range: TimeRange) -> bool:
        return any(
            SegmentEventIntersection(
                Segment(time_range), sweep.best().detections
            ).num_events_in_seg
            > 0
            for sweep in self.threshold_sweeps
        )

    def post_plot(
        self, time_range: TimeRange, input_ax: Axes, extra_axes: Sequence[Axes]
    ):
        tups = zip(self.threshold_sweeps, extra_axes, self.titles, self.colors)
        for sweep, ax, title, color in tups:
            ax.hlines(
                sweep.best().threshold,
                *time_range,
                clip_on=False,
                linestyles="dashed"
            )
            add_event_arrows(ax, sweep.best().correct_detections, color="green")
            add_event_arrows(ax, sweep.best().incorrect_detections, color="red")
            ax.text(0.05, 0.91, title, color=color, transform=ax.transAxes)