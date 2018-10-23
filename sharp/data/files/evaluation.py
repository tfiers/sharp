from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.data.files.stdlib import DictFile


class ThresholdSweepFile(DictFile):
    extension = ".eval" + DictFile.extension

    def read(self) -> ThresholdSweep:
        return super().read()

    def write(self, sweep: ThresholdSweep):
        output = {
            te.threshold: {
                "detections": te.detections,
                "reference_segs": te.reference_segs,
            }
            for te in sweep.threshold_evaluations
        }
        super().write(output)
