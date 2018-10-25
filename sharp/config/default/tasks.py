from typing import Sequence

from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.plot.misc.gevec_principle import PlotGEVecPrinciple
from sharp.tasks.plot.results.base import MultiEnvelopeFigureMaker
from sharp.tasks.plot.results.weights import PlotWeights
from sharp.tasks.plot.results.envelopes import PlotEnvelopes
from sharp.tasks.plot.misc.reference import PlotReferenceMaker
from sharp.tasks.plot.results.PR_and_latency import PlotLatencyAndPR
from sharp.tasks.plot.results.latency_scatter import PlotLatencyScatter
from sharp.tasks.plot.misc.data_summary import PlotRecordingSummaries
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF, SaveBPFinfo


def multi_envelope_plots(**em_kwargs) -> Sequence[MultiEnvelopeFigureMaker]:
    return (
        PlotWeights(**em_kwargs),
        PlotEnvelopes(**em_kwargs),
        PlotLatencyScatter(**em_kwargs),
        PlotLatencyAndPR(**em_kwargs),
        PlotLatencyAndPR(zoom_from=0.65, **em_kwargs),
    )


tasks_to_run = (
    PlotGEVecPrinciple(),
    PlotReferenceMaker(),
    PlotRecordingSummaries(),
    SaveBPFinfo(),
    *multi_envelope_plots(
        subdir="LSM-main",
        envelope_makers=(
            ApplyOnlineBPF(),
            SpatiotemporalConvolution(num_delays=0),
            SpatiotemporalConvolution(num_delays=1),
        ),
    ),
    *multi_envelope_plots(
        subdir="space-time-comp",
        envelope_makers=(
            SpatiotemporalConvolution(num_delays=0),
            SpatiotemporalConvolution(num_delays=1),
            SpatiotemporalConvolution(num_delays=2),
            SpatiotemporalConvolution(num_delays=3),
            SpatiotemporalConvolution(num_delays=5),
            SpatiotemporalConvolution(num_delays=10),
            SpatiotemporalConvolution(num_delays=20),
        ),
    ),
)
