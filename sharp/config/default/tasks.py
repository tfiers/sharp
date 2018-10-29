from typing import Sequence

from sharp.tasks.plot.results.PR_and_latency import PlotLatencyAndPR
from sharp.tasks.plot.results.base import MultiEnvelopeFigureMaker
from sharp.tasks.plot.results.envelopes import PlotEnvelopes
from sharp.tasks.plot.results.latency_scatter import PlotLatencyScatter
from sharp.tasks.plot.results.searchgrid.PR import PR
from sharp.tasks.plot.results.searchgrid.latency import Latency
from sharp.tasks.plot.results.weights import PlotWeights


def multi_envelope_plots(**em_kwargs) -> Sequence[MultiEnvelopeFigureMaker]:
    return (
        PlotWeights(**em_kwargs),
        PlotEnvelopes(**em_kwargs),
        PlotLatencyScatter(**em_kwargs),
        PlotLatencyAndPR(**em_kwargs),
        PlotLatencyAndPR(zoom_from=0.65, **em_kwargs),
    )


def searchgrids(**kwargs):
    return (PR(**kwargs), Latency(**kwargs))


tasks_to_run = (
    # PlotGEVecPrinciple(),
    # PlotReferenceMaker(),
    # PlotRecordingSummaries(),
    # SaveBPFinfo(),
    # *multi_envelope_plots(
    #     subdir="LSM-main",
    #     envelope_makers=(
    #         ApplyOnlineBPF(),
    #         SpatiotemporalConvolution(num_delays=0),
    #         SpatiotemporalConvolution(num_delays=1),
    #     ),
    # ),
    *searchgrids(subdir="space-time-comp"),
)
