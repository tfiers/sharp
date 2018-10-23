from logging import getLogger

import numpy as np
from luigi import FloatParameter, IntParameter

from sharp.data.files.config import intermediate_output_dir
from sharp.data.files.evaluation import ThresholdSweepFile
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.tasks.base import SharpTask, TaskParameter
from sharp.tasks.evaluate.threshold import evaluate_threshold
from sharp.tasks.signal.base import EnvelopeMaker, InputDataMixin

log = getLogger(__name__)


class ThresholdSweeper(SharpTask, InputDataMixin):
    """
    Calculate performance of a detector based on its output envelope, for
    different detection thresholds.
    """

    envelope_maker: EnvelopeMaker = TaskParameter()

    num_thresholds = IntParameter()
    lockout_percentile = FloatParameter(25)

    def requires(self):
        return self.input_data_makers + (self.envelope_maker,)

    def output(self) -> ThresholdSweepFile:
        filename = self.envelope_maker.output().stem
        return ThresholdSweepFile(
            intermediate_output_dir / "threshold-sweeps", filename
        )

    @property
    def lockout_time(self) -> float:
        return np.percentile(
            self.reference_segs_all.duration, self.lockout_percentile
        )

    def run(self):
        sweep = ThresholdSweep()
        threshold_range = self.envelope_maker.envelope_test.range
        log.info(
            f"Evaluating {self.num_thresholds} thresholds, "
            f"with a lockout time of {1000 * self.lockout_time:.3g} ms."
        )
        while len(sweep.thresholds) < self.num_thresholds:
            threshold = sweep.get_next_threshold(threshold_range)
            new_threshold_evaluation = evaluate_threshold(
                self.envelope_maker.envelope_test,
                threshold,
                self.lockout_time,
                self.reference_segs_test,
            )
            sweep.add_threshold_evaluation(new_threshold_evaluation)

        self.output().write(sweep)
