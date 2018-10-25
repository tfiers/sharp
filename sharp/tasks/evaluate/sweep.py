from logging import getLogger

import numpy as np

from sharp.config.params import intermediate_output_dir, main_config
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

    output_root = intermediate_output_dir / "threshold-sweeps"

    envelope_maker: EnvelopeMaker = TaskParameter()

    def requires(self):
        return self.input_data_makers + (self.envelope_maker,)

    def output(self) -> ThresholdSweepFile:
        return ThresholdSweepFile(
            directory=self.output_root / self.envelope_maker.output_subdir,
            filename=self.envelope_maker.output_filename,
        )

    @property
    def lockout_time(self) -> float:
        return np.percentile(
            self.reference_segs_all.duration, main_config.lockout_percentile
        )

    def run(self):
        sweep = ThresholdSweep()
        threshold_range = self.envelope_maker.envelope_test.range
        log.info(
            f"Evaluating {main_config.num_thresholds} thresholds, "
            f"with a lockout time of {1000 * self.lockout_time:.3g} ms."
        )
        while len(sweep.thresholds) < main_config.num_thresholds:
            threshold = sweep.get_next_threshold(threshold_range)
            new_threshold_evaluation = evaluate_threshold(
                self.envelope_maker.envelope_test,
                threshold,
                self.lockout_time,
                self.reference_segs_test,
            )
            sweep.add_threshold_evaluation(new_threshold_evaluation)

        self.output().write(sweep)
