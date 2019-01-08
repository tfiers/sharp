def multi_envelope_plots(**em_kwargs):
    from sharp.tasks.plot.results.PR_and_latency import PlotLatencyAndPR
    from sharp.tasks.plot.results.envelopes import PlotEnvelopes
    from sharp.tasks.plot.results.latency_scatter import PlotLatencyScatter
    from sharp.tasks.plot.results.latency_info import WriteLatencyInfo

    return (
        # PlotWeights(**em_kwargs),
        PlotLatencyScatter(**em_kwargs),
        WriteLatencyInfo(**em_kwargs),
        #
        PlotEnvelopes(**em_kwargs),
        PlotLatencyAndPR(**em_kwargs),
        PlotLatencyAndPR(zoom_from=0.70, **em_kwargs),
    )


def searchgrids(**kwargs):
    from sharp.tasks.plot.results.searchgrid.PR import PlotPRGrid
    from sharp.tasks.plot.results.searchgrid.latency import PlotLatencyGrid

    return (PlotPRGrid(**kwargs), PlotLatencyGrid(**kwargs))


def tasks_on_hold():
    from sharp.tasks.plot.misc.data_summary import PlotRecordingSummaries
    from sharp.tasks.plot.misc.gevec_principle import PlotGEVecPrinciple
    from sharp.tasks.plot.misc.reference import PlotReferenceMaker
    from sharp.tasks.plot.misc.offline_steps import PlotOfflineStepsMultifig
    from sharp.tasks.text.offline_steps_info import WriteOfflineInfo
    from sharp.tasks.text.online_BPF_info import WriteOnlineBPFInfo
    from sharp.tasks.text.evaluation_info import WriteEvalInfo
    from sharp.tasks.plot.results.searchlines.GEVec import PlotSearchLines_GEVec
    from sharp.data.hardcoded.filters.best import ProposedOnlineBPF
    from sharp.data.hardcoded.filters.literature import (
        DuttaReplica,
        EgoStengelReplica,
        FalconReplica,
    )
    from sharp.tasks.signal.online_bpf import ApplyOnlineBPF
    from sharp.tasks.plot.misc.F_score import PlotIsoFlines

    return (
        PlotGEVecPrinciple(),
        PlotReferenceMaker(),
        PlotRecordingSummaries(),
        *searchgrids(subdir="space-time-comp"),
        WriteOnlineBPFInfo(),
        WriteOfflineInfo(),
        WriteEvalInfo(),
        PlotSearchLines_GEVec(),
        PlotOfflineStepsMultifig(),
        *multi_envelope_plots(
            subdir="online-BPF",
            envelope_makers=(
                ApplyOnlineBPF(ripple_filter=EgoStengelReplica()),
                ApplyOnlineBPF(ripple_filter=DuttaReplica()),
                ApplyOnlineBPF(ripple_filter=FalconReplica()),
                ApplyOnlineBPF(ripple_filter=ProposedOnlineBPF()),
            ),
        ),
        *multi_envelope_plots(
            subdir="LSM-main",
            envelope_makers=(
                ApplyOnlineBPF(),
                SpatiotemporalConvolution(num_delays=0),
                SpatiotemporalConvolution(num_delays=1),
            ),
        ),
        PlotIsoFlines(),
    )


from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF
from sharp.tasks.neuralnet.apply import ApplyRNN
from sharp.tasks.plot.misc.training import PlotValidLoss

tasks_to_run = (
    # *tasks_on_hold(),
    *multi_envelope_plots(
        subdir="conclusion",
        envelope_makers=(
            ApplyOnlineBPF(),
            SpatiotemporalConvolution(num_delays=10),
            # SpatiotemporalConvolution(num_delays=10, channel_combo_name="pyr"),
            ApplyRNN(),
        ),
    ),
    PlotValidLoss(),
)
