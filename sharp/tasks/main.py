from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.neuralnet.apply import ApplyRNN
from sharp.tasks.plot.signals.reference import PlotReferenceMaker
from sharp.tasks.plot.wrapper import PlotEvaluations
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF, SaveBPFinfo

# fmt: off
poster_CNSN = PlotEvaluations(
    combi_ID="CNSN-poster",
    envelope_makers=(
        ApplyOnlineBPF(),
        ApplyRNN(),
    )
)

chapter_multi_lin = PlotEvaluations(
    combi_ID="chapter-multi-lin",
    envelope_makers=(
        ApplyOnlineBPF(),
        SpatiotemporalConvolution(num_delays=0),
        SpatiotemporalConvolution(num_delays=1),
        # SpatiotemporalConvolution(num_delays=2),
        # SpatiotemporalConvolution(num_delays=10),
    ),
)

TASKS_TO_RUN = (
    # PlotReferenceMaker(),
    # PlotValidLoss(),
    SaveBPFinfo(),
    chapter_multi_lin,
)
# fmt: on
