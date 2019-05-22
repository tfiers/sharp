def minipaper():
    from sharp.tasks.neuralnet.apply import ApplyRNN
    from sharp.tasks.plot.misc.training import PlotValidLoss
    from sharp.tasks.plot.paper.PR_curve import Plot_PR_Curve
    from sharp.tasks.plot.paper.grid import AccuracyGrid, LatencyGrid
    from sharp.tasks.plot.paper.latency import PlotLatency
    from sharp.tasks.plot.paper.signals import PlotSignals
    from sharp.tasks.signal.online_bpf import ApplyOnlineBPF

    return (
        PlotValidLoss(),
        PlotSignals(),
        Plot_PR_Curve(),
        PlotLatency(),
        AccuracyGrid(envelope_maker=ApplyRNN()),
        AccuracyGrid(envelope_maker=ApplyOnlineBPF()),
        LatencyGrid(envelope_maker=ApplyRNN()),
        LatencyGrid(envelope_maker=ApplyOnlineBPF()),
    )


def get_default_tasks():
    # from sharp.tasks.signal.downsample import DownsampleAllRecordings
    # return DownsampleAllRecordings()
    from sharp.tasks.plot.stat.vignette import PlotVignettes
    from sharp.config.load import config

    return (PlotVignettes(file_ID=f) for f in config.raw_data)
