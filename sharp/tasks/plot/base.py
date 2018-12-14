from logging import getLogger
from typing import Iterable, Sequence, Tuple

from luigi import BoolParameter
from matplotlib import pyplot as plt, style
from matplotlib.axes import Axes
from matplotlib.pyplot import close
from numpy import ceil, diff, ones

from sharp.config.load import config, final_output_dir, output_root
from sharp.data.files.figure import FigureTarget
from sharp.data.hardcoded.style import paperfig, symposium
from sharp.data.types.aliases import subplots
from sharp.data.types.signal import Signal
from sharp.tasks.base import SharpTask
from sharp.tasks.plot.util.annotations import add_segments
from sharp.tasks.plot.util.scalebar import (
    add_scalebar,
    add_time_scalebar,
    add_voltage_scalebar,
)
from sharp.tasks.plot.util.signal import plot_signal
from sharp.tasks.signal.base import InputDataMixin

log = getLogger(__name__)


class FigureMaker(SharpTask):
    output_dir = final_output_dir

    def __init__(self, *args, **kwargs):
        close("all")
        style.use(symposium)
        super().__init__(*args, **kwargs)


TimeRange = Tuple[float, float]


class TimeRangesPlotter(FigureMaker, InputDataMixin):
    """
    For each time-range out of a set of time-ranges, makes a figure with
    multiple synchronized signals plotted beneath each other.
    
    By default, makes many plots that together cover the entire test slice.
    """

    selected_time_ranges_only: bool = BoolParameter(default=True)
    reference_channel_only: bool = BoolParameter(default=True)
    full_range_scalebars: bool = False

    window_size: float = 0.6
    # Duration of each time slice (and thus of each plot). In seconds.

    def requires(self):
        return self.input_data_makers

    def output(self):
        for start, stop in self.time_ranges:
            filename = f"{start:.1f}--{stop:.1f}"
            yield FigureTarget(self.output_dir, filename)

    def work(self):
        for time_range, output in zip(self.time_ranges, self.output()):
            fig = self.make_figure(time_range)
            log.info(f"Saving target {output.relative_to(output_root)}")
            output.write(fig)
            plt.close()

    @property
    def time_ranges(self):
        if self.selected_time_ranges_only:
            return config.time_ranges
        else:
            return self.all_time_ranges

    @property
    def input_signal(self):
        if self.reference_channel_only:
            return self.reference_channel_test
        else:
            return self.multichannel_test

    @property
    def all_time_ranges(self) -> Iterable[TimeRange]:
        duration = self.multichannel_test.duration
        num_ranges = int(ceil(duration / self.window_size))
        start = 0
        for i in range(num_ranges):
            stop = start + self.window_size
            time_range = (start, min(stop, duration))
            if self.is_included(time_range):
                yield time_range
            start = stop

    def is_included(self, time_range: TimeRange) -> bool:
        """ Whether to plot the given time range, or to skip it. """
        return True

    def make_figure(self, time_range):
        nrows = 1 + len(self.extra_signals)
        axheights = ones(nrows)
        if not self.reference_channel_only:
            num_channels = self.multichannel_test.num_channels
            axheights[0] = 1 + num_channels / 6
        figheight = 0.25 + 0.10 * len(self.extra_signals)
        fig, axes = subplots(
            nrows=nrows,
            sharex=True,
            figsize=paperfig(width=0.55, height=0.9 * figheight),
            gridspec_kw=dict(height_ratios=axheights),
        )
        input_ax = axes[0]
        extra_axes = axes[1:]
        self.plot_input_signal(time_range, input_ax)
        self.plot_other_signals(time_range, extra_axes)
        self.post_plot(time_range, input_ax, extra_axes)
        add_segments(input_ax, self.reference_segs_test)
        add_time_scalebar(
            extra_axes[0],
            select_scalebar_time(time_range),
            "ms",
            pos_along=0.73,
            pos_across=1.25,
            in_layout=False,
        )
        fig.tight_layout(rect=(0.02, 0, 1, 1))
        return fig

    def plot_input_signal(self, time_range: TimeRange, ax: Axes):
        plot_signal_neat(
            self.input_signal,
            time_range,
            ax,
            tight_ylims=self.selected_time_ranges_only,
        )
        add_voltage_scalebar(ax, 500, "uV", pos_along=0.1, pos_across=-0.01)

    def plot_other_signals(self, time_range: TimeRange, axes: Sequence[Axes]):
        for ax, signal, color in zip(axes, self.extra_signals, self.colors):
            plot_signal_neat(
                signal,
                time_range,
                ax,
                color=color,
                tight_ylims=self.selected_time_ranges_only,
            )
            if self.full_range_scalebars:
                add_scalebar(
                    ax,
                    direction="v",
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


def plot_signal_neat(
    signal: Signal,
    time_range: TimeRange,
    ax: Axes,
    time_grid=False,
    y_grid=False,
    clip_on=False,
    **kwargs,
):
    """
    An opinionated `plot_signal()`, that yields a clean plot, well fitted for
    plotting multiple signals in an array of axes.
    """
    plot_signal(
        signal,
        time_range,
        ax=ax,
        time_grid=time_grid,
        y_grid=y_grid,
        clip_on=clip_on,
        **kwargs,
    )
    # Add some padding in the beginning:
    start, end = time_range
    time_span = end - start
    ax.set_xlim(start - 0.03 * time_span, end)


def select_scalebar_time(time_range: TimeRange) -> int:
    tspan = diff(time_range)
    if tspan < 0.2:
        bar = 10
    elif tspan < 1:
        bar = 50
    else:
        bar = 200
    return bar
