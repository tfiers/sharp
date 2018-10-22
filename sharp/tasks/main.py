from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.plot.misc.gevec_principle import PlotGEVecPrinciple
from sharp.tasks.plot.misc.weights import PlotWeights
from sharp.tasks.plot.signals.envelopes import PlotEnvelopes
from sharp.tasks.plot.signals.reference import PlotReferenceMaker
from sharp.tasks.plot.summary.PR_and_latency import PlotLatencyAndPR
from sharp.tasks.plot.summary.latency_scatter import PlotLatencyScatter
from sharp.tasks.plot.summary.recording import PlotRecordingSummaries
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF, SaveBPFinfo

em_kwargs = dict(
    envelope_makers=(
        ApplyOnlineBPF(),
        SpatiotemporalConvolution(num_delays=0),
        SpatiotemporalConvolution(num_delays=1),
    )
)

multi_envelope_plotters = (
    PlotWeights(**em_kwargs),
    PlotEnvelopes(**em_kwargs),
    PlotLatencyScatter(**em_kwargs),
    PlotLatencyAndPR(**em_kwargs),
    PlotLatencyAndPR(zoom_from=0.65, **em_kwargs),
)


TASKS_TO_RUN = (
    PlotGEVecPrinciple(),
    PlotReferenceMaker(),
    PlotRecordingSummaries(),
    SaveBPFinfo(),
    *multi_envelope_plotters,
)
