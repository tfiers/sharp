from sharp.tasks.plot.misc.searchlines import BPF_SearchLines_Mixin
from sharp.tasks.plot.results.searchlines.base import PlotEnvelopeSearchLines
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF


class PlotSearchLines_BPF(PlotEnvelopeSearchLines, BPF_SearchLines_Mixin):
    @property
    def envelope_maker_lists(self):
        return [
            [
                ApplyOnlineBPF(ripple_filter=ripple_filter, order=order)
                for order in self.orders
            ]
            for ripple_filter in self.filters.values()
        ]
