from logging import getLogger

from sharp.config.load import config, intermediate_output_dir
from sharp.data.files.evaluation import ThresholdSweepFile
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.data.types.split import TrainTestSplit
from sharp.tasks.base import SharpTask, TaskParameter
from sharp.tasks.evaluate.threshold import evaluate_threshold
from sharp.tasks.signal.base import EnvelopeMaker
from sharp.tasks.signal.reference import MakeReference

log = getLogger(__name__)


class ThresholdSweeper(SharpTask):
    """
    Calculate performance of a detector based on its output envelope, for
    different detection thresholds.
    """

    output_root = intermediate_output_dir / "threshold-sweeps"

    envelope_maker: EnvelopeMaker = TaskParameter()
    reference_maker: MakeReference = TaskParameter(default=MakeReference())

    def requires(self):
        return (self.envelope_maker, self.reference_maker)

    def output(self) -> ThresholdSweepFile:
        return ThresholdSweepFile(
            directory=(
                self.output_root
                / self.reference_maker.output_filename
                / self.envelope_maker.output_subdir
            ),
            filename=self.envelope_maker.output_filename,
        )

    @property
    def lockout_time(self) -> float:
        # return percentile(
        #     self.reference_segs_all.duration, config.lockout_percentile
        # )
        return config.lockout_time

    def work(self):
        sweep = ThresholdSweep()
        threshold_range = self.envelope_maker.envelope_test.range
        log.info(
            f"Evaluating {config.num_thresholds} thresholds, "
            f"with a lockout time of {1000 * self.lockout_time:.3g} ms."
        )
        refsegs_all = self.reference_maker.output().read()
        refsegs_test = TrainTestSplit(
            self.envelope_maker.envelope, refsegs_all
        ).segments_test
        while len(sweep.thresholds) < config.num_thresholds:
            threshold = sweep.get_next_threshold(threshold_range)
            new_threshold_evaluation = evaluate_threshold(
                self.envelope_maker.envelope_test,
                threshold,
                self.lockout_time,
                refsegs_test,
            )
            sweep.add_threshold_evaluation(new_threshold_evaluation)

        self.output().write(sweep)
