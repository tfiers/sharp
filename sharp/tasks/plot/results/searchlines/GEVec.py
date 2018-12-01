from numpy.core.multiarray import arange

from sharp.data.types.style import blue
from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.plot.results.searchlines.base import PlotSearchLines


class PlotSearchLines_GEVec(PlotSearchLines):
    filename = "GEVec"
    num_delays = arange(40)
    # reference_color = red
    plot_IQR = True
    convolvers = tuple(
        SpatiotemporalConvolution(num_delays=num) for num in num_delays
    )
    titles = ["GEVec, all channels"]
    colors = [blue]
    envelope_maker_lists = [convolvers]
