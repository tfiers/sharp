from typing import Dict

from luigi import Parameter
from numpy.core.multiarray import arange

from sharp.config.filters import LTIRippleFilter, num_delays_BPF
from sharp.tasks.base import CustomParameter
from sharp.tasks.plot.results.searchlines.base import PlotSearchLines
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF


class PlotSearchLines_BPF(PlotSearchLines):

    filename = Parameter()
    filters: Dict[str, LTIRippleFilter] = CustomParameter()
    orders = arange(14)
    num_delays = num_delays_BPF(orders)

    @property
    def titles(self):
        return list(self.filters.keys())

    @property
    def envelope_maker_lists(self):
        return [
            [
                ApplyOnlineBPF(ripple_filter=ripple_filter, order=order)
                for order in self.orders
            ]
            for ripple_filter in self.filters.values()
        ]
