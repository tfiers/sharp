from itertools import product

from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.plot.results.base import MultiEnvelopeFigureMaker


class SearchGrid(MultiEnvelopeFigureMaker):

    subdir = 'space-time-comp'

    num_delays = (0, 1, 2, 3, 5, 10, 20)

    args = product(num_delays, )

    envelope_makers = [SpatiotemporalConvolution()]
