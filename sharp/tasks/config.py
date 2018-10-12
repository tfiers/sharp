def get_tasks_to_run():
    from sharp.tasks.evaluate.algorithms import EvaluateAlgorithms
    from sharp.tasks.plot.signals.detectors import PlotDetectors
    from sharp.tasks.plot.signals.reference import PlotReferenceMaker
    from sharp.tasks.plot.summary.PR_tradeoff import PlotPR
    from sharp.tasks.plot.summary.latency import PlotLatency
    from sharp.tasks.plot.summary.training import PlotValidLoss
    from sharp.tasks.plot.summary.latency_scatter import PlotLatencyScatter
    from sharp.tasks.plot.summary.PR_tradeoff import CONTINUOUS, DISCRETE

    return (
        EvaluateAlgorithms(),
        # PlotLatencyScatter(),
        # PlotPR(
        #     start_proportion=0, line_kwargs=CONTINUOUS, figsize_multiplier=0.7
        # ),
        # PlotPR(
        #     start_proportion=0.65, line_kwargs=DISCRETE, ticks_topright=True
        # ),
        # PlotLatency(),
        # PlotValidLoss(),
        # PlotDetectors(),
        # PlotReferenceMaker(),
    )
