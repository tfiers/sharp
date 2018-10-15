from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.neuralnet.apply import ApplyRNN
from sharp.tasks.plot.base import MultiEnvelopeFigureMaker
from sharp.tasks.plot.signals.reference import PlotReferenceMaker
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF

# fmt: off
CNSN_poster = MultiEnvelopeFigureMaker(
    combi_ID="CNSN-poster",
    envelope_makers=(
        ApplyOnlineBPF(),
        ApplyRNN(),
    )
)

chapter_multi_lin = MultiEnvelopeFigureMaker(
    combi_ID="chapter-multi-lin",
    envelope_makers=(
        ApplyOnlineBPF(),
        SpatiotemporalConvolution(delays=0),
        SpatiotemporalConvolution(delays=3),
    ),
)

TASKS_TO_RUN = (
    PlotReferenceMaker(),
    chapter_multi_lin,
)
# fmt: on
