from typing import Sequence

from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.data.types.signal import Signal
from sharp.tasks.base import SharpTask, TaskListParameter
from sharp.tasks.evaluate.sweep import ThresholdSweeper
from sharp.tasks.signal.base import EnvelopeMaker
from sharp.util import cached


class MultiEnvelopeEvaluator(SharpTask):

    envelope_makers: Sequence[EnvelopeMaker] = TaskListParameter()

    def requires(self):
        return self.threshold_sweepers

    @property
    def threshold_sweepers(self) -> Sequence[ThresholdSweeper]:
        return [
            ThresholdSweeper(envelope_maker=em) for em in self.envelope_makers
        ]

    @property
    @cached
    def threshold_sweeps(self) -> Sequence[ThresholdSweep]:
        return [sweeper.output().read() for sweeper in self.threshold_sweepers]

    @property
    def test_envelopes(self) -> Sequence[Signal]:
        return [em.envelope_test for em in self.envelope_makers]
