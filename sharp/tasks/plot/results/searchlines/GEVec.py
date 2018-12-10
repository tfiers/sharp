from numpy.core.multiarray import arange

from sharp.data.hardcoded.style import blue
from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.plot.results.searchlines.base import PlotEnvelopeSearchLines


class PlotSearchLines_GEVec(PlotEnvelopeSearchLines):
    filename = "GEVec"
    num_delays = arange(40)
    legend_outside = False
    # reference_color = red
    plot_IQR = True
    convolvers = tuple(
        SpatiotemporalConvolution(num_delays=num) for num in num_delays
    )
    labels = ["GEVec, all channels"]
    colors = [blue]
    envelope_maker_lists = [convolvers]
