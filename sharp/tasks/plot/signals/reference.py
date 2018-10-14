from sharp.tasks.plot.signals.base import TimeRangesPlotter


class PlotReferenceMaker(TimeRangesPlotter):
    """
    Plots the input signal, the reference maker filter output, and the
    reference segments. Does this only for the evaluation slice.
    """

    @property
    def output_dir(self):
        return super().output_dir / "reference"

    @property
    def extra_signals(self):
        return [self._reference_maker.envelope]

    def post_plot(self, time_range, input_ax, extra_axes):
        ax = extra_axes[0]
        ax.hlines(self._reference_maker.threshold_high, *time_range)
        ax.hlines(
            self._reference_maker.threshold_low, *time_range, linestyles="dashed"
        )
