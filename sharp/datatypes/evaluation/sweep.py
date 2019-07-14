from typing import List, Tuple

import numpy as np

from fileflow import Saveable
from sharp.datatypes.base import HDF5File
from sharp.datatypes.evaluation.threshold import ThresholdEvaluation
from sharp.datatypes.segments import SegmentArray


def vectorizing_property(name: str):
    """
    A property that automatically gathers a scalar measure from different
    threshold evaluations.

    :param name:  A scalar attribute of a ThresholdEvaluation instance.
    """

    def vectorize_attribute(self: "ThresholdSweep") -> np.ndarray:
        """
        Gather a scalar measure from different threshold evaluations.
        """
        values = [
            getattr(threshold_evaluation, name)
            for threshold_evaluation in self.threshold_evaluations
        ]
        return np.array(values)

    return property(fget=vectorize_attribute)


class ThresholdSweep(Saveable):

    """
    Evaluates the performance of an SWR detection algorithm, based on its
    output envelope, for a range of different detection thresholds on the
    envelope.
    """

    def get_filetype():
        return ThresholdSweepFile

    threshold_evaluations: List[ThresholdEvaluation]
    # Always ordered from highest to lowest threshold.

    def __init__(self):
        self.threshold_evaluations = []

    # FYI: We do not make this a proper dataclass, as `dataclass` does not
    # recognize the following properties as properties, and takes them as init
    # args (which they should not be).
    threshold: np.ndarray = vectorizing_property("threshold")
    num_detected: np.ndarray = vectorizing_property("num_detected")
    num_undetected: np.ndarray = vectorizing_property("num_undetected")
    num_correct: np.ndarray = vectorizing_property("num_correct")
    num_incorrect: np.ndarray = vectorizing_property("num_incorrect")
    recall: np.ndarray = vectorizing_property("recall")
    precision: np.ndarray = vectorizing_property("precision")
    FDR: np.ndarray = vectorizing_property("FDR")
    F1: np.ndarray = vectorizing_property("F1")
    F2: np.ndarray = vectorizing_property("F2")
    abs_delays_median: np.ndarray = vectorizing_property("abs_delays_median")
    rel_delays_median: np.ndarray = vectorizing_property("rel_delays_median")
    rel_delays_Q1: np.ndarray = vectorizing_property("rel_delays_Q1")
    rel_delays_Q3: np.ndarray = vectorizing_property("rel_delays_Q3")

    thresholds: np.ndarray = threshold
    # Alias, for readability.

    def add_threshold_evaluation(self, new: ThresholdEvaluation):
        """
        Inserts the given object into `self.threshold_evaluations`, such that
        `self.thresholds` stays sorted from high to low.
        """
        if len(self.thresholds) == 0:
            insertion_index = 0
        elif new.threshold < min(self.thresholds):
            insertion_index = len(self.thresholds)
        else:
            insertion_index = np.argmin(new.threshold < self.thresholds)
        self.threshold_evaluations.insert(insertion_index, new)

    def get_next_threshold(self, threshold_range: Tuple[float, float]) -> float:
        """
        Succesive calls yield a sequence of thresholds within `threshold_range`
        that cover the [0, 1] domains of `recall` and `precision` well. Assumes
        a reasonably smooth PR-curve.

        (Tries to eliminate gaps in the number of detected reference SWR
        segments and the number of correct detections between different
        thresholds).
        """
        if len(self.thresholds) == 0:
            return max(threshold_range)
        elif len(self.thresholds) == 1:
            return min(threshold_range)
        else:
            num_correct_gap = np.abs(np.diff(self.num_correct))
            num_detected_gap = np.abs(np.diff(self.num_detected))
            index_largest_nc_gap = np.argmax(num_correct_gap)
            index_largest_nd_gap = np.argmax(num_detected_gap)
            largest_nc_gap = num_correct_gap[index_largest_nc_gap]
            largest_nd_gap = num_detected_gap[index_largest_nd_gap]
            if largest_nc_gap > largest_nd_gap:
                index_gap = index_largest_nc_gap
            else:
                index_gap = index_largest_nd_gap
            thresholds_at_gap = self.thresholds[index_gap : index_gap + 2]
            return np.mean(thresholds_at_gap)


class ThresholdSweepFile(HDF5File):
    def write_to_file(self, sweep: ThresholdSweep, f):
        for i, te in enumerate(sweep.threshold_evaluations):
            group = f.create_group(name=str(i))
            group.attrs["threshold"] = te.threshold
            group.create_dataset("first_detections", data=te.first_detections)
            group.create_dataset(
                "correct_detections", data=te.correct_detections
            )
            group.create_dataset(
                "incorrect_detections", data=te.incorrect_detections
            )
            group.create_dataset(
                "detected_reference_segs",
                data=te.detected_reference_segs.asarray(),
            )
            group.create_dataset(
                "undetected_reference_segs",
                data=te.undetected_reference_segs.asarray(),
            )

    def read_from_file(self, f) -> ThresholdSweep:
        sweep = ThresholdSweep()
        for group in f.values():
            te = ThresholdEvaluation(
                threshold=group.attrs["threshold"],
                first_detections=group["first_detections"][()],
                correct_detections=group["correct_detections"][()],
                incorrect_detections=group["incorrect_detections"][()],
                detected_reference_segs=SegmentArray(
                    group["detected_reference_segs"][()], check=False
                ),
                undetected_reference_segs=SegmentArray(
                    group["undetected_reference_segs"][()], check=False
                ),
            )
            sweep.add_threshold_evaluation(te)
        return sweep
