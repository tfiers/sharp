from typing import Sequence

from sharp.data.types.aliases import Axes
from sharp.tasks.plot.signals.base import TimeRange, TimeRangesPlotter
from sharp.tasks.plot.util.annotations import add_events


class PlotDetectors(TimeRangesPlotter):
    """
    Plots synched time slices of the input signal and the algorithm output
    envelopes.

    Annotates these time series plots with reference segments and detection
    times of all algorithms.
    """

    @property
    def output_dir(self):
        return super().output_dir / "output-signals"

    @property
    def extra_signals(self):
        return self.evaluation.envelopes

    def post_plot(
        self, time_range: TimeRange, input_ax: Axes, extra_axes: Sequence[Axes]
    ):
        for sweep, ax in zip(self.evaluation.threshold_sweeps, extra_axes):
            te = sweep.best
            ax.hlines(
                sweep.best.threshold,
                *time_range,
                clip_on=False,
                linestyles="dashed"
            )
            add_events(ax, sweep.best.correct_detections, color="C2")
            add_events(ax, sweep.best.incorrect_detections, color="C1")
