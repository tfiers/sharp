from typing import Sequence, Tuple

from sharp.datatypes.evaluation import ThresholdSweep, PerformanceMatrix
from sharp.datatypes.segments import SegmentArray
from sharp.datatypes.signal import Signal


def evaluate_online_performance(
    envelope: Signal, reference_SWR_segs: SegmentArray
) -> ThresholdSweep:
    ...


def distill_perf_matrix(
    mult_detect_matrix: Sequence[Tuple[float, float]],
    ORF_performances: Sequence[ThresholdSweep],
    RNN_performances: Sequence[ThresholdSweep],
) -> PerformanceMatrix:
    ...
