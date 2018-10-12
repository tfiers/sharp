from sharp.tasks.plot.signals.base import TimeRangesPlotter
from sharp.tasks.signal.reference import MakeReference


class PlotReferenceMaker(TimeRangesPlotter):
    """
    Plots the input signal, the reference maker filter output, and the
    reference segments. Does this only for the evaluation slice.
    """

    reference_maker = MakeReference()

    @property
    def output_dir(self):
        return super().output_dir / "reference"

    @property
    def extra_signals(self):
        return [self.reference_maker.envelope]

    def post_plot(self, time_range, input_ax, extra_axes):
        ax = extra_axes[0]
        ax.hlines(self.reference_maker.threshold_high, *time_range)
        ax.hlines(
            self.reference_maker.threshold_low, *time_range, linestyles="dashed"
        )
