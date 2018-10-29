from luigi import Parameter
from matplotlib.ticker import PercentFormatter

from sharp.tasks.evaluate.multi_envelope import MultiEnvelopeEvaluator
from sharp.tasks.plot.base import FigureMaker


fraction_formatter = PercentFormatter(xmax=1, decimals=0)


class ResultsFigureMaker(FigureMaker):

    output_dir = FigureMaker.output_dir / "results"


class MultiEnvelopeFigureMaker(ResultsFigureMaker, MultiEnvelopeEvaluator):

    subdir = Parameter()

    @property
    def output_dir(self):
        return super().output_dir / self.subdir

    @property
    def colors(self):
        return [f"C{i}" for i in range(len(self.envelope_makers))]

    @property
    def titles(self):
        return [em.title for em in self.envelope_makers]
