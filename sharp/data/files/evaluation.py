from numpy import array

from fklab.segments import Segment
from sharp.data.files.stdlib import DictFile
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.data.types.evaluation.threshold import ThresholdEvaluation
from sharp.data.types.intersection import SegmentEventIntersection


class ThresholdSweepFile(DictFile):
    extension = ".eval" + DictFile.extension

    def write(self, sweep: ThresholdSweep):
        data_dicts = [
            {
                "threshold": float(te.threshold),
                "detections": te.detections.tolist(),
                "reference_segs": te.reference_segs._data.tolist(),
            }
            for te in sweep.threshold_evaluations
        ]
        # TOML cannot have a list as root object.
        obj = {"threshold_evaluations": data_dicts}
        super().write(obj)

    def read(self) -> ThresholdSweep:
        sweep = ThresholdSweep()
        obj = super().read()
        for data_dict in obj["threshold_evaluations"]:
            detections = array(data_dict["detections"])
            reference_segs = Segment(data_dict["reference_segs"])
            te = ThresholdEvaluation(
                detections=detections,
                reference_segs=reference_segs,
                intersection=SegmentEventIntersection(
                    reference_segs, detections
                ),
                threshold=data_dict["threshold"],
            )
            sweep.add_threshold_evaluation(te)
        return sweep
