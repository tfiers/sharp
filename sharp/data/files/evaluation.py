from h5py import File as HDF5File, Group

from fklab.segments import Segment
from sharp.data.files.base import FileTarget
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.data.types.evaluation.threshold import ThresholdEvaluation
from sharp.data.types.intersection import SegmentEventIntersection
from sharp.util import cached


class ThresholdSweepFile(FileTarget):
    extension = ".hdf5"

    def write(self, sweep: ThresholdSweep):
        with HDF5File(self.path_string, "w") as file:
            # reference_segs are the same for each ThresholdEvaluation
            file.create_dataset(
                "reference_segs", data=sweep.best().reference_segs._data
            )
            group = file.create_group("detections")
            for i, te in enumerate(sweep.threshold_evaluations):
                dataset = group.create_dataset(name=str(i), data=te.detections)
                dataset.attrs["threshold"] = te.threshold

    @cached
    def read(self) -> ThresholdSweep:
        with HDF5File(self.path_string, "r") as file:
            reference_segs = Segment(file["reference_segs"][:])
            sweep = ThresholdSweep()
            group: Group = file["detections"]
            for dataset in group.values():
                detections = dataset[:]
                te = ThresholdEvaluation(
                    detections=detections,
                    reference_segs=reference_segs,
                    intersection=SegmentEventIntersection(
                        reference_segs, detections
                    ),
                    threshold=dataset.attrs["threshold"],
                )
                sweep.add_threshold_evaluation(te)
        return sweep
