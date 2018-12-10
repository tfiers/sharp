from sharp.data.types.split import TrainTestSplit
from sharp.tasks.plot.base import TimeRangesPlotter
from sharp.tasks.signal.reference import MakeReference


class PlotReferenceMaker(TimeRangesPlotter):
    """
    Plots the input signal, the reference maker filter output, and the
    reference segments. Does this only for the evaluation slice.
    """

    selected_time_ranges_only = False
    reference_channel_only = False
    full_range_scalebars = True
    output_dir = TimeRangesPlotter.output_dir / "reference"

    reference_maker = MakeReference()

    def requires(self):
        return (self.reference_maker,) + super().requires()

    colors = ["black"]

    @property
    def extra_signals(self):
        envelope = self.reference_maker.envelope
        envelope_test = TrainTestSplit(envelope).signal_test
        return [envelope_test]

    def post_plot(self, time_range, input_ax, extra_axes):
        ax = extra_axes[0]
        ax.hlines(self.reference_maker.threshold_high, *time_range)
        ax.hlines(
            self.reference_maker.threshold_low, *time_range, linestyles="dashed"
        )
