from logging import getLogger
from typing import Tuple

import numpy as np
from luigi import FloatParameter, IntParameter
from sharp.data.types.evaluation import ThresholdSweep
from sharp.data.files.evaluation import ThresholdSweepFile
from sharp.data.files.config import output_root
from sharp.tasks.evaluate.slice import EvaluationSliceMaker
from sharp.tasks.evaluate.threshold import evaluate_threshold

log = getLogger(__name__)


class ThresholdSweeper(EvaluationSliceMaker):
    """
    Calculate performance of a detector based on its output envelope, for
    different detection thresholds.
    """

    num_thresholds = IntParameter()
    recall_best = FloatParameter(0.90)
    lockout_percentile = FloatParameter(25)

    def output(self) -> ThresholdSweepFile:
        filename = self.requires().output().stem
        return ThresholdSweepFile(output_root / "threshold-sweeps", filename)

    @property
    def lockout_time(self) -> float:
        return np.percentile(
            self.reference_segs.duration, self.lockout_percentile
        )

    def run(self):
        sweep = ThresholdSweep()
        threshold_range = self.envelope.range
        log.info(f"Evaluating {self.num_thresholds} thresholds")
        while len(sweep.thresholds) < self.num_thresholds:
            threshold = get_next_threshold(sweep, threshold_range)
            new_threshold_evaluation = evaluate_threshold(
                self.envelope, threshold, self.lockout_time, self.reference_segs
            )
            index = get_insertion_index(sweep, threshold)
            sweep.threshold_evaluations.insert(index, new_threshold_evaluation)
        sweep.recall_best = self.recall_best
        self.output().write(sweep)


def get_insertion_index(sweep: ThresholdSweep, new_threshold: float) -> int:
    """
    Index into `sweep.thresholds` such that this array stays sorted from high
    to low when the new threshold is inserted there.
    """
    if len(sweep.thresholds) == 0:
        insertion_index = 0
    elif new_threshold < np.min(sweep.thresholds):
        insertion_index = len(sweep.thresholds)
    else:
        insertion_index = np.argmin(new_threshold < sweep.thresholds)
    return insertion_index


def get_next_threshold(
    sweep: ThresholdSweep, threshold_range: Tuple[float, float]
) -> float:
    """
    Succesive calls yield a sequence of thresholds that cover the [0, 1]
    domains of `sweep.recall` and `sweep.precision` well.
    """
    if len(sweep.thresholds) == 0:
        return max(threshold_range)
    elif len(sweep.thresholds) == 1:
        return min(threshold_range)
    else:
        R_gap = np.abs(np.diff(sweep.recall))
        P_gap = np.abs(np.diff(sweep.precision))
        index_largest_R_gap = np.argmax(R_gap)
        index_largest_P_gap = np.argmax(P_gap)
        largest_R_gap = R_gap[index_largest_R_gap]
        largest_P_gap = P_gap[index_largest_P_gap]
        if largest_R_gap > largest_P_gap:
            index = index_largest_R_gap
        else:
            index = index_largest_P_gap
        surrounding_thresholds = sweep.thresholds[index : index + 2]
        return np.mean(surrounding_thresholds)
