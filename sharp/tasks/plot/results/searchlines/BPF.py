from numpy.core.multiarray import arange

from sharp.tasks.plot.results.searchlines.base import PlotSearchLines
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF, online_BPFs


class PlotSearchLines_BPF(PlotSearchLines):
    filename = "online-BPFs"
    orders = arange(14)
    num_delays = orders
    titles = list(online_BPFs.keys())

    @property
    def envelope_maker_lists(self):
        return [
            [ApplyOnlineBPF(filter_name=filter_name, N=N) for N in self.orders]
            for filter_name in self.titles
        ]
