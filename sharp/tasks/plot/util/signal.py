from typing import Optional, Tuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import arange, ndarray

from sharp.data.types.aliases import subplots
from sharp.data.types.signal import Signal
from sharp.tasks.signal.util import time_to_index


def plot_signal(
    signal: Signal,
    time_range: Tuple[float, float],
    y_scale: float = 500,
    height: float = 0.5,
    channels: Optional[ndarray] = None,
    bottom_first: bool = True,
    tight_ylims: bool = False,
    zero_lines: bool = True,
    time_grid: bool = True,
    y_grid: Optional[bool] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> (Figure, Axes):
    """
    Plot a time-slice of a single- or multichannel signal.

    When the signal is multichannel, each channel will be plotted with the same
    scale.

    :param time_range:  Time slice to plot. In seconds.
    :param y_scale:  How much data-y-units the visual vertical spacing between
                channels represents.
    :param height:  Height of each channel, in inches.
    :param channels:  Which channels to plot. Plots all channels by default.
    :param bottom_first:  If True (default), the first channel will be
                plotted at the bottom of the figure.
    :param tight_ylims:  If True, adapts the ylims to tightly fit the data
                visible in `time_range`. If False (default), makes sure that
                plots of different time ranges of `signal` will all have the
                same ylims.
    :param zero_lines:  Whether to plot grey y==0 lines in each channel.
    :param time_grid:  Whether to plot vertical gridlines, with corresponding
                absolute time ticks and ticklabels.
    :param y_grid:  Whether to plot horizontal gridlines, with corresponding
                y-ticks and -ticklabels. By default, only plots a y-grid if the
                signal is single-channel and `zero_lines` is False.
    :param ax:  The axes to plot on. If None (default), creates a new figure and
                axes.
    :param kwargs:  Passed on to `ax.plot()`.
    """
    signal = signal.as_matrix()
    if ax is None:
        fig, ax = subplots(figsize=(12, height * signal.num_channels))
    else:
        fig = ax.get_figure()
    if y_grid is None:
        if signal.num_channels == 1 and not zero_lines:
            y_grid = True
        else:
            y_grid = False
    if channels is None:
        channels = arange(signal.num_channels)
    if ("color" not in kwargs) and ("c" not in kwargs):
        kwargs["color"] = "black"
    ix = time_to_index(
        time_range, signal.fs, arr_size=signal.num_samples, clip=True
    )
    y: Signal = signal[slice(*ix), channels]
    t = y.get_time_vector(t0=time_range[0])
    if bottom_first:
        y_offsets = y_scale * arange(0, signal.num_channels)
    else:
        y_offsets = y_scale * arange(0, -signal.num_channels, -1)
    y_separated = y + y_offsets
    if zero_lines:
        ax.hlines(y_offsets, *time_range, colors="grey", lw=1)
    ax.plot(t, y_separated, **kwargs)
    ax.set_xlim(time_range)
    if not tight_ylims:
        ax.set_ylim(_get_global_ylims(signal, y_scale))
    if time_grid:
        ax.set_xlabel("Time (s)")
    else:
        ax.grid(False, which="x")
        ax.set_xticks([])
    if not y_grid:
        ax.grid(False, which="y")
        ax.set_yticks([])
    return (fig, ax)


def _get_global_ylims(signal: Signal, spacing: float):
    return (
        signal[:, 0].min(),
        signal[:, -1].max() + spacing * (signal.num_channels - 1),
    )
