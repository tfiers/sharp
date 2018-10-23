from sharp.tasks.evaluate.multi_envelope import MultiEnvelopeEvaluator
from sharp.tasks.plot.base import FigureMaker


class ResultsFigureMaker(FigureMaker):

    output_dir = FigureMaker.output_dir / "results"


class MultiEnvelopeFigureMaker(ResultsFigureMaker, MultiEnvelopeEvaluator):
    @property
    def colors(self):
        return [f"C{i}" for i in range(len(self.envelope_makers))]

    @property
    def titles(self):
        return [em.title for em in self.envelope_makers]
