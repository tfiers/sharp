from logging import getLogger
from typing import Type

import numpy as np
from luigi import FloatParameter, IntParameter, TaskParameter

from sharp.data.files.config import output_root
from sharp.data.files.evaluation import ThresholdSweepFile
from sharp.data.types.evaluation import ThresholdSweep
from sharp.tasks.base import SharpTask
from sharp.tasks.evaluate.threshold import evaluate_threshold
from sharp.tasks.signal.base import EnvelopeMaker

log = getLogger(__name__)


class ThresholdSweeper(SharpTask):
    """
    Calculate performance of a detector based on its output envelope, for
    different detection thresholds.
    """

    envelope_maker_class: Type[EnvelopeMaker] = TaskParameter()
    num_thresholds = IntParameter()
    recall_best = FloatParameter(0.90)
    lockout_percentile = FloatParameter(25)

    @property
    def envelope_maker(self):
        return self.envelope_maker_class()

    def requires(self):
        return self.envelope_maker

    def output(self) -> ThresholdSweepFile:
        filename = self.envelope_maker.output().stem
        return ThresholdSweepFile(output_root / "threshold-sweeps", filename)

    @property
    def lockout_time(self) -> float:
        return np.percentile(
            self.envelope_maker.reference_segs_train.duration, self.lockout_percentile
        )

    def run(self):
        sweep = ThresholdSweep()
        threshold_range = self.envelope_maker.envelope_test.range
        log.info(f"Evaluating {self.num_thresholds} thresholds")
        while len(sweep.thresholds) < self.num_thresholds:
            threshold = sweep.get_next_threshold(threshold_range)
            new_threshold_evaluation = evaluate_threshold(
                self.envelope_maker.envelope_test,
                threshold,
                self.lockout_time,
                self.envelope_maker.reference_maker.reference_segs_test,
            )
            sweep.add_threshold_evaluation(new_threshold_evaluation)

        sweep.recall_best = self.recall_best
        self.output().write(sweep)
