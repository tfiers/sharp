from typing import Optional, Sequence

from matplotlib.axes import Axes
from sharp.config.load import config


def draw_channelmap(
    ax: Axes, active_channels: Optional[Sequence[int]] = None, ms=5.5
):
    ax.invert_yaxis()  # Origin: top-left
    ax.set_aspect("equal")
    ax.axis("off")
    # Close the path:
    probe_outline = tuple(config.probe_outline) + (config.probe_outline[0],)
    probe_x = [vertex[0] for vertex in probe_outline]
    probe_y = [vertex[1] for vertex in probe_outline]
    ax.plot(probe_x, probe_y, c="black", lw=1.5)
    num_channels = len(config.electrodes_x)
    all_channels = tuple(range(num_channels))
    if active_channels is None:
        active_channels = all_channels
    for channel in all_channels:
        if channel in active_channels:
            style = dict(color="black")
        else:
            style = dict(color="grey", markerfacecolor="none")
        ax.plot(
            config.electrodes_x[channel],
            config.electrodes_y[channel],
            marker="o",
            ms=ms,
            **style,
        )
