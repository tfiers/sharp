from sharp.tasks.plot.base import MultiEnvelopeFigureMaker
from sharp.tasks.plot.signals.envelopes import PlotEnvelopes
from sharp.tasks.plot.summary.PR_tradeoff import CONTINUOUS, DISCRETE, PlotPR
from sharp.tasks.plot.summary.latency import PlotLatency
from sharp.tasks.plot.summary.latency_scatter import PlotLatencyScatter


class PlotEvaluations(MultiEnvelopeFigureMaker):
    """ Wrapper task """

    def requires(self):
        kwargs = dict(
            combi_ID=self.combi_ID, envelope_makers=self.envelope_makers
        )
        return (
            PlotLatency(**kwargs),
            PlotLatencyScatter(**kwargs),
            PlotPR(
                start_proportion=0,
                line_kwargs=CONTINUOUS,
                figsize_multiplier=0.7,
                **kwargs,
            ),
            PlotPR(
                start_proportion=0.65,
                line_kwargs=DISCRETE,
                ticks_topright=True,
                **kwargs,
            ),
            PlotEnvelopes(**kwargs),
        )
