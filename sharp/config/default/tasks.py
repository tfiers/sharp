from sharp.tasks.plot.misc.approx_lit_BPF import ApproximateLiteratureBPF


def multi_envelope_plots(**em_kwargs):
    from sharp.tasks.plot.results.PR_and_latency import PlotLatencyAndPR
    from sharp.tasks.plot.results.envelopes import PlotEnvelopes
    from sharp.tasks.plot.results.latency_scatter import PlotLatencyScatter
    from sharp.tasks.plot.results.weights import PlotWeights

    return (
        PlotWeights(**em_kwargs),
        PlotEnvelopes(**em_kwargs),
        PlotLatencyScatter(**em_kwargs),
        PlotLatencyAndPR(**em_kwargs),
        PlotLatencyAndPR(zoom_from=0.65, **em_kwargs),
    )


def searchgrids(**kwargs):
    from sharp.tasks.plot.results.searchgrid.PR import PlotPRGrid
    from sharp.tasks.plot.results.searchgrid.latency import PlotLatencyGrid

    return (PlotPRGrid(**kwargs), PlotLatencyGrid(**kwargs))


def tasks_on_hold():
    from sharp.tasks.multilin.apply import SpatiotemporalConvolution
    from sharp.tasks.plot.misc.F_score import PlotIsoFlines
    from sharp.tasks.plot.misc.data_summary import PlotRecordingSummaries
    from sharp.tasks.plot.misc.gevec_principle import PlotGEVecPrinciple
    from sharp.tasks.plot.misc.reference import PlotReferenceMaker
    from sharp.tasks.signal.online_bpf import ApplyOnlineBPF
    from sharp.tasks.plot.misc.offline_steps import PlotOfflineStepsMultifig
    from sharp.tasks.text.offline_steps_info import WriteOfflineInfo
    from sharp.tasks.text.online_BPF_info import SaveBPFinfo
    from sharp.tasks.text.evaluation_info import WriteEvalInfo
    from sharp.tasks.plot.results.searchlines.GEVec import PlotSearchLines_GEVec
    from sharp.config.default.filters import (
        main_comp,
        cheby1_comp,
        cheby2_comp,
        sinc_FIR_comp,
    )
    from sharp.tasks.plot.misc.filter_theory_searchlines import (
        PlotFilterTheorySearchlines,
    )
    from sharp.tasks.plot.results.searchlines.BPF import PlotSearchLines_BPF

    return (
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
        *searchgrids(subdir="space-time-comp"),
        PlotIsoFlines(),
        PlotOfflineStepsMultifig(),
        WriteOfflineInfo(),
        WriteEvalInfo(),
        PlotSearchLines_GEVec(),
        PlotFilterTheorySearchlines(**main_comp),
        PlotSearchLines_BPF(**main_comp),
        PlotSearchLines_BPF(**sinc_FIR_comp),
        PlotSearchLines_BPF(**cheby1_comp),
        PlotSearchLines_BPF(**cheby2_comp),
    )


tasks_to_run = (
    # *tasks_on_hold(),
    ApproximateLiteratureBPF(),
)
