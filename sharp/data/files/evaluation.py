from fklab.segments import Segment
from sharp.data.files.base import HDF5Target
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.data.types.evaluation.threshold import ThresholdEvaluation
from sharp.util.misc import cached


class ThresholdSweepFile(HDF5Target):
    def write(self, sweep: ThresholdSweep):
        with self.open_file_for_write() as file:
            for i, te in enumerate(sweep.threshold_evaluations):
                group = file.create_group(name=str(i))
                group.attrs["threshold"] = te.threshold
                group.create_dataset(
                    "first_detections", data=te.first_detections
                )
                group.create_dataset(
                    "correct_detections", data=te.correct_detections
                )
                group.create_dataset(
                    "incorrect_detections", data=te.incorrect_detections
                )
                group.create_dataset(
                    "detected_reference_segs",
                    data=te.detected_reference_segs._data,
                )
                group.create_dataset(
                    "undetected_reference_segs",
                    data=te.undetected_reference_segs._data,
                )

    @cached
    def read(self) -> ThresholdSweep:
        sweep = ThresholdSweep()
        with self.open_file_for_read() as file:
            for group in file.values():
                te = ThresholdEvaluation(
                    threshold=group.attrs["threshold"],
                    first_detections=group["first_detections"][:],
                    correct_detections=group["correct_detections"][:],
                    incorrect_detections=group["incorrect_detections"][:],
                    detected_reference_segs=Segment(
                        group["detected_reference_segs"][:], check=False
                    ),
                    undetected_reference_segs=Segment(
                        group["undetected_reference_segs"][:], check=False
                    ),
                )
                sweep.add_threshold_evaluation(te)
        return sweep
