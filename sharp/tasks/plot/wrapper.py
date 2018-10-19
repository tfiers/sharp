from sharp.tasks.plot.base import MultiEnvelopeFigureMaker
from sharp.tasks.plot.signals.envelopes import PlotEnvelopes
from sharp.tasks.plot.summary.PR_and_latency import DISCRETE, PlotLatencyAndPR
from sharp.tasks.plot.summary.latency_scatter import PlotLatencyScatter
from sharp.tasks.plot.summary.weights import PlotWeights


class PlotEvaluations(MultiEnvelopeFigureMaker):
    """ Wrapper task """

    def requires(self):
        kwargs = dict(
            combi_ID=self.combi_ID, envelope_makers=self.envelope_makers
        )
        return (
            PlotWeights(**kwargs)
            # PlotEnvelopes(**kwargs),
            # PlotLatencyScatter(**kwargs),
            # PlotLatencyAndPR(**kwargs),
            # PlotLatencyAndPR(zoom_from=0.65, line_kwargs=DISCRETE, **kwargs),
        )
