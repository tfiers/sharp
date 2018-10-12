from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from sharp.data.types.aliases import Axes, Figure, IndexList
from sharp.data.types.signal import Signal
from sharp.tasks.signal.util import time_to_index


def plot_signal(
    signal: Signal,
    time_range: Tuple[float, float],
    spacing: float = 500,
    height: float = 0.5,
    channels: Optional[IndexList] = None,
    reverse_channels: bool = True,
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

    :param time_range: Time slice to plot. In seconds.
    :param spacing: Vertical spacing between the zero lines of channels, in
        data units.
    :param height: Height of each channel, in inches.
    :param channels: Which channels to plot. Plots all channels by default.
    :param reverse_channels: If True (default), the first channel will be
        plotted at the bottom of the figure.
    :param zero_lines: Whether to plot grey y==0 lines in each channel.
    :param time_grid: Whether to plot vertical gridlines, with corresponding
        absolute time ticks and ticklabels.
    :param y_grid: Whether to plot horizontal gridlines, with corresponding
        y-ticks and -ticklabels. By default, only plots a y-grid if the signal
        is single-channel and `zero_lines` is False.
    :param ax: The axes to plot on. If None (default), creates a new figure and
        axes.
    :param kwargs: Passed on to `ax.plot()`.
    """
    signal = signal.as_matrix()
    N, C = signal.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, height * C))
    else:
        fig = ax.get_figure()
    if y_grid is None:
        if C == 1 and not zero_lines:
            y_grid = True
        else:
            y_grid = False
    if channels is None:
        channels = np.arange(C)
    if ("color" not in kwargs) and ("c" not in kwargs):
        kwargs["color"] = "black"
    ix = time_to_index(time_range, signal.fs, arr_size=N, clip=True)
    y: Signal = signal[slice(*ix), channels]
    t = y.get_time_vector(t0=time_range[0])
    if reverse_channels:
        y_offsets = spacing * np.arange(0, C)
    else:
        y_offsets = spacing * np.arange(0, -C, -1)
    y_separated = y + y_offsets
    if zero_lines:
        ax.hlines(y_offsets, *time_range, colors="grey", lw=1)
    ax.plot(t, y_separated, **kwargs)
    ax.set_xlim(time_range)
    if time_grid:
        ax.set_xlabel("Time (s)")
    else:
        ax.grid(False, which="x")
        ax.set_xticks([])
    if not y_grid:
        ax.grid(False, which="y")
        ax.set_yticks([])
    return (fig, ax)
