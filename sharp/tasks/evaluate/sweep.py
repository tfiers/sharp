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
from sharp.tasks.signal.split import TrainTestSplitter

log = getLogger(__name__)


class ThresholdSweeper(SharpTask):
    """
    Calculate performance of a detector based on its output envelope, for
    different detection thresholds.
    """

    envelope_maker: Type[EnvelopeMaker] = TaskParameter()
    num_thresholds = IntParameter()
    recall_best = FloatParameter(0.90)
    lockout_percentile = FloatParameter(25)

    @property
    def data(self):
        return TrainTestSplitter(envelope_maker=self.envelope_maker)

    def requires(self):
        return self.data

    def output(self) -> ThresholdSweepFile:
        filename = self.data.envelope_maker.output().stem
        return ThresholdSweepFile(output_root / "threshold-sweeps", filename)

    @property
    def lockout_time(self) -> float:
        return np.percentile(
            self.data.train.reference_segs.duration, self.lockout_percentile
        )

    def run(self):
        sweep = ThresholdSweep()
        threshold_range = self.data.test.envelope.range
        log.info(f"Evaluating {self.num_thresholds} thresholds")
        while len(sweep.thresholds) < self.num_thresholds:
            threshold = sweep.get_next_threshold(threshold_range)
            new_threshold_evaluation = evaluate_threshold(
                self.data.test.envelope,
                threshold,
                self.lockout_time,
                self.data.test.reference_segs,
            )
            sweep.add_threshold_evaluation(new_threshold_evaluation)

        sweep.recall_best = self.recall_best
        self.output().write(sweep)
