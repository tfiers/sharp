from numpy import array

from fklab.segments import Segment
from sharp.data.files.stdlib import DictFile
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.data.types.evaluation.threshold import ThresholdEvaluation
from sharp.data.types.intersection import SegmentEventIntersection
from sharp.util import cached


class ThresholdSweepFile(DictFile):
    extension = ".eval" + DictFile.extension

    def write(self, sweep: ThresholdSweep):
        te_dicts = [
            {
                "threshold": float(te.threshold),
                "detections": te.detections.tolist(),
            }
            for te in sweep.threshold_evaluations
        ]
        # reference_segs are the same for each ThresholdEvaluation
        obj = {
            "reference_segs": sweep.best().reference_segs._data.tolist(),
            "threshold_evaluations": te_dicts,
        }
        super().write(obj)

    @cached
    def read(self) -> ThresholdSweep:
        sweep = ThresholdSweep()
        obj = super().read()
        reference_segs = Segment(obj["reference_segs"])
        for te_dict in obj["threshold_evaluations"]:
            detections = array(te_dict["detections"])
            te = ThresholdEvaluation(
                detections=detections,
                reference_segs=reference_segs,
                intersection=SegmentEventIntersection(
                    reference_segs, detections
                ),
                threshold=te_dict["threshold"],
            )
            sweep.add_threshold_evaluation(te)
        return sweep
