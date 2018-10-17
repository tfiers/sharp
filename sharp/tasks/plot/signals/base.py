from logging import getLogger
from typing import Iterable, Sequence, Tuple

from luigi import FloatParameter
from matplotlib import pyplot as plt
from numpy import ones
from numpy.core.umath import ceil

from sharp.data.files.config import output_root
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import Axes, Figure
from sharp.data.types.signal import Signal
from sharp.data.types.split import TrainTestSplit
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.plot.util.annotations import add_segments
from sharp.tasks.plot.util.scalebar import (
    add_scalebar,
    add_time_scalebar,
    add_voltage_scalebar,
)
from sharp.tasks.plot.util.signal import plot_signal
from sharp.tasks.signal.base import InputDataMixin

TimeRange = Tuple[float, float]

log = getLogger(__name__)


class TimeRangesPlotter(FigureMaker, InputDataMixin):
    """
    Makes many plots of a set of time ranges, that together cover the entire
    test slice.
    """

    window_size: float = FloatParameter(0.6, significant=False)
    # Duration of each time slice (and thus of each plot). In seconds.

    def requires(self):
        return self.input_data_makers

    def output(self):
        for start, stop in self.time_ranges:
            filename = f"{start:.1f}--{stop:.1f}"
            yield FigureTarget(self.output_dir, filename)

    def run(self):
        for time_range, output in zip(self.time_ranges, self.output()):
            fig = self.make_figure(time_range)
            log.info(f"Saving target {output.relative_to(output_root)}")
            output.write(fig)
            plt.close()

    @property
    def time_ranges(self) -> Iterable[TimeRange]:
        num_ranges = int(
            ceil(self.multichannel_test.duration / self.window_size)
        )
        split = TrainTestSplit(self.reference_channel_full)
        eval_start, eval_stop = split.time_range_test
        start = eval_start
        for i in range(num_ranges):
            stop = start + self.window_size
            time_range = (start, min(stop, eval_stop))
            if self.include(time_range):
                yield time_range
            start = stop

    def include(self, time_range: TimeRange) -> bool:
        """ Whether to plot the given time range, or to skip it. """
        return True

    def make_figure(self, time_range):
        nrows = 1 + len(self.extra_signals)
        num_channels = self.multichannel_test.num_channels
        axheights = ones(nrows)
        axheights[0] = 1 + num_channels / 6
        fig, axes = plt.subplots(
            nrows=nrows,
            sharex=True,
            figsize=(5, 1 + sum(axheights)),
            gridspec_kw=dict(height_ratios=axheights),
        )  # type: Figure, Sequence[Axes]
        input_ax = axes[0]
        extra_axes = axes[1:]
        self.plot_input_signal(time_range, input_ax)
        self.plot_other_signals(time_range, extra_axes)
        self.post_plot(time_range, input_ax, extra_axes)
        add_segments(input_ax, self.reference_segs_test)
        add_time_scalebar(
            extra_axes[0], 100, "ms", pos_along=0.74, pos_across=1.1
        )
        fig.tight_layout(rect=(0.02, 0, 1, 1))
        return fig

    def plot_input_signal(self, time_range: TimeRange, ax: Axes):
        plot_clean_sig(self.multichannel_test, time_range, ax)
        add_voltage_scalebar(ax, 1, "mV", pos_along=0.1, pos_across=0)

    def plot_other_signals(self, time_range: TimeRange, axes: Sequence[Axes]):
        for ax, signal in zip(axes, self.extra_signals):
            plot_clean_sig(signal, time_range, ax)
            ax.set_ylim(signal.range)
            add_scalebar(
                ax,
                orient="v",
                length=signal.span,
                label="",
                pos_along=0,
                pos_across=0,
            )

    def post_plot(
        self, time_range: TimeRange, input_ax: Axes, extra_axes: Sequence[Axes]
    ):
        """
        Optional hook to override. Called after all signals have been plotted.
        """
        pass

    @property
    def extra_signals(self) -> Sequence[Signal]:
        """ Signals to plot (besides the input signal). """
        return []


def plot_clean_sig(signal: Signal, time_range: TimeRange, ax: Axes):
    plot_signal(
        signal,
        time_range,
        ax=ax,
        time_grid=False,
        y_grid=False,
        lw=1.2,
        clip_on=False,
    )
    # Add some padding in the beginning:
    start, end = time_range
    time_span = end - start
    ax.set_xlim(start - 0.03 * time_span, end)
