from typing import Sequence, Tuple

from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.data.types.signal import Signal
from sharp.tasks.base import SharpTask, TaskListParameter
from sharp.tasks.evaluate.sweep import ThresholdSweeper
from sharp.tasks.signal.base import EnvelopeMaker
from sharp.util.misc import cached


class MultiEnvelopeEvaluator(SharpTask):

    envelope_makers: Sequence[EnvelopeMaker] = TaskListParameter()

    def requires(self):
        return self.threshold_sweepers

    @property
    def threshold_sweepers(self) -> Tuple[ThresholdSweeper, ...]:
        return tuple(
            ThresholdSweeper(envelope_maker=em) for em in self.envelope_makers
        )

    @property
    @cached
    def threshold_sweeps(self) -> Tuple[ThresholdSweep, ...]:
        return tuple(
            sweeper.output().read() for sweeper in self.threshold_sweepers
        )

    @property
    def test_envelopes(self) -> Tuple[Signal, ...]:
        return tuple(em.envelope_test for em in self.envelope_makers)
