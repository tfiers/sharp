from logging import getLogger

from luigi import IntParameter
from numpy import abs
from scipy.signal import lfilter

from sharp.data.hardcoded.filters.base import LTIRippleFilter
from sharp.data.hardcoded.filters.literature import FalconReplica
from sharp.data.types.signal import Signal
from sharp.tasks.base import TaskParameter
from sharp.tasks.signal.base import EnvelopeMaker

log = getLogger(__name__)


class ApplyOnlineBPF(EnvelopeMaker):

    ripple_filter: LTIRippleFilter = TaskParameter(default=FalconReplica())
    order: int = IntParameter(default=None)

    @property
    def title(self):
        return self.ripple_filter.title

    output_subdir = "online-BPF"

    @property
    def output_filename(self):
        return f"{self.ripple_filter}, N={self.order}"

    def work(self):
        fs = self.input_signal.fs
        self.ripple_filter.order = self.order
        self.ripple_filter.fs = fs
        b, a = self.ripple_filter.tf
        filtered = lfilter(b, a, self.input_signal)
        envelope = abs(filtered)
        self.output().write(Signal(envelope, fs))

    @property
    def input_signal(self):
        return self.reference_channel_full.as_vector()
