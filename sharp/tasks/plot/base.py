from typing import Sequence

from luigi.task import Parameter
from matplotlib import style

from sharp.data.files.config import output_root
from sharp.data.types.evaluation import ThresholdSweep
from sharp.data.types.signal import Signal
from sharp.tasks.base import SharpTask, TaskListParameter
from sharp.tasks.evaluate.sweep import ThresholdSweeper
from sharp.tasks.plot.style import symposium
from sharp.tasks.signal.base import EnvelopeMaker


class FigureMaker(SharpTask):
    output_dir = output_root / "figures"

    def __init__(self, *args, **kwargs):
        style.use(symposium)
        super().__init__(*args, **kwargs)


class MultiEnvelopeFigureMaker(FigureMaker):

    combi_ID: str = Parameter()
    envelope_makers: Sequence[EnvelopeMaker] = TaskListParameter()

    @property
    def output_dir(self):
        return super().output_dir / self.combi_ID

    def requires(self):
        return self.threshold_sweepers

    @property
    def threshold_sweepers(self) -> Sequence[ThresholdSweeper]:
        return [
            ThresholdSweeper(envelope_maker=em) for em in self.envelope_makers
        ]

    @property
    def threshold_sweeps(self) -> Sequence[ThresholdSweep]:
        return [sweeper.output().read() for sweeper in self.threshold_sweepers]

    @property
    def test_envelopes(self) -> Sequence[Signal]:
        return [em.envelope_test for em in self.envelope_makers]

    @property
    def titles(self):
        return [em.title for em in self.envelope_makers]

    @property
    def colors(self):
        return [f"C{i}" for i in range(len(self.envelope_makers))]
