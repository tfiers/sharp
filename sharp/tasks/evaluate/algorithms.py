from typing import Dict, NewType, Sequence, Tuple, TypeVar

from sharp.data.types.evaluation import ThresholdSweep
from sharp.data.types.signal import Signal
from sharp.tasks.base import SharpTask
from sharp.tasks.evaluate.sweep import ThresholdSweeper
from sharp.tasks.neuralnet.apply import ApplyRNN
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF

T = TypeVar("T")
SweeperID = NewType("SweeperID", str)


class EvaluateAlgorithms(SharpTask):
    sweeper_order: Tuple[SweeperID, ...] = ("sota", "proposal")

    def sort_values_by_sweeper(self, dictt: Dict[SweeperID, T]) -> Sequence[T]:
        return [dictt[key] for key in self.sweeper_order]

    @property
    def threshold_sweepers(self) -> Sequence[ThresholdSweeper]:
        return self.sort_values_by_sweeper(
            {
                "sota": ThresholdSweeper(envelope_maker=ApplyOnlineBPF),
                "proposal": ThresholdSweeper(envelope_maker=ApplyRNN),
            }
        )

    def requires(self) -> Dict[SweeperID, ThresholdSweeper]:
        return self.threshold_sweepers

    @property
    def threshold_sweeps(self) -> Sequence[ThresholdSweep]:
        return [sweeper.output().read() for sweeper in self.threshold_sweepers]

    @property
    def envelopes(self) -> Sequence[Signal]:
        return [sweeper.envelope for sweeper in self.threshold_sweepers]
