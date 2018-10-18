from sharp.tasks.plot.base import MultiEnvelopeFigureMaker
from sharp.tasks.plot.summary.PR_and_latency import PlotLatencyAndPR


class PlotEvaluations(MultiEnvelopeFigureMaker):
    """ Wrapper task """

    def requires(self):
        kwargs = dict(
            combi_ID=self.combi_ID, envelope_makers=self.envelope_makers
        )
        return (
            # PlotLatency(**kwargs),
            # PlotLatencyScatter(**kwargs),
            PlotLatencyAndPR(
                zoom_from=0,
                # figsize_multiplier=0.7,
                **kwargs,
            ),
            # PlotPR(
            #     zoom_from=0.65,
            #     line_kwargs=DISCRETE,
            #     ticks_topright=True,
            #     **kwargs,
            # ),
            # PlotEnvelopes(**kwargs),
        )
