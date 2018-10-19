from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.core.multiarray import ndarray

from fklab.segments import Segment
from sharp.data.types.aliases import BooleanArray, EventList
from sharp.data.types.intersection import SegmentEventIntersection


@dataclass
class ThresholdEvaluation:
    """
    Classifies detections into {correct, incorrect},
    and reference segments into {detected, not_detected}.

    Calculates detection delays and summarising performance measures based on
    these classifications.
    """

    #
    # ------
    # Inputs
    # ------

    detections: EventList
    # When the SWR detector fired, in seconds.

    reference_segs: Segment
    # The (start, stop) tuples that indicate baseline "true" SWR segments. In
    # seconds.

    intersection: SegmentEventIntersection
    # Pre-calculated comparison between `detections` and `reference_segs`.

    threshold: float
    # (For vectorization in ThresholdSweep).

    #
    # ------
    # Recall
    # ------

    @property
    def _reference_seg_is_detected(self) -> BooleanArray:
        return self.intersection.num_events_in_seg > 0

    @property
    def detected_reference_segs(self) -> EventList:
        return self.reference_segs[self._reference_seg_is_detected]

    @property
    def undetected_reference_segs(self) -> EventList:
        return self.reference_segs[~self._reference_seg_is_detected]

    @property
    def recall(self) -> float:
        return len(self.detected_reference_segs) / len(self.reference_segs)

    #
    # ---------
    # Precision
    # ---------

    @property
    def _detection_is_correct(self) -> BooleanArray:
        return self.intersection.event_is_in_seg

    @property
    def correct_detections(self) -> EventList:
        """ Detection events that hit a reference segment. """
        return self.detections[self._detection_is_correct]

    @property
    def incorrect_detections(self) -> EventList:
        """ False positive detections. """
        return self.detections[~self._detection_is_correct]

    @property
    def precision(self) -> float:
        return len(self.correct_detections) / len(self.detections)

    @property
    def FDR(self) -> float:
        """ False discovery rate. """
        return 1 - self.precision

    #
    # --------
    # F-scores
    # --------

    @property
    def F1(self):
        return self.get_F_score(1)

    @property
    def F2(self):
        return self.get_F_score(2)

    def get_F_score(self, beta):
        """
        Weighted harmonic mean of recall and precision.

        "The Fβ-score was derived so that "Fβ" measures the effectiveness
        of retrieval with respect to a user who attaches β times as much
        importance to recall as precision.""
        """
        denominator = (beta ** 2 * self.precision) + self.recall
        if denominator == 0:
            return 0
        else:
            numerator = (1 + beta ** 2) * self.precision * self.recall
            return numerator / denominator

    #
    # ----------------
    # Detection delays
    # ----------------

    @property
    def first_detections(self) -> EventList:
        """
        For each detected reference segment, the time of the first `detection`
        that caught it.
        """
        return self.detections[self.intersection.first_event_in_seg]

    @property
    def abs_delays(self) -> ndarray:
        """
        Absolute detection delays. One data point per detected reference
        segment. Positive if the detection happened after the start of the
        reference segment.
        """
        return self.first_detections - self.detected_reference_segs.start

    @property
    def rel_delays(self) -> ndarray:
        """
        Detection delays, as a fraction of the duration of the corresponding
        reference segments. One data point per detected reference segment.
        """
        return self.abs_delays / self.detected_reference_segs.duration


def vectorized_property(name: str):
    """
    :param name: Name of a scalar attribute of a ThresholdEvaluation instance.
    """

    def vectorize_attribute(self: "ThresholdSweep") -> ndarray:
        """
        Gather a scalar measure from different threshold evaluations.
        """
        values = [
            getattr(threshold_evaluation, name)
            for threshold_evaluation in self.threshold_evaluations
        ]
        return np.array(values)

    return property(vectorize_attribute)


class ThresholdSweep:

    threshold_evaluations: List[ThresholdEvaluation]
    # Always ordered from highest to lowest threshold.

    def __init__(self):
        self.threshold_evaluations = []

    recall_best: Optional[float] = None
    # At which approximate recall value the `best` threshold should be chosen.
    # If not specified, chooses the threshold with maximal F2-score.

    threshold: ndarray = vectorized_property("threshold")
    recall: ndarray = vectorized_property("recall")
    precision: ndarray = vectorized_property("precision")
    FDR: ndarray = vectorized_property("FDR")
    F1: ndarray = vectorized_property("F1")
    F2: ndarray = vectorized_property("F2")

    thresholds: ndarray = threshold
    # Alias, for readability.

    @property
    def AUC(self) -> float:
        """ Area under precision-recall curve. """
        if len(self.threshold_evaluations) > 0:
            return np.trapz(self.precision, self.recall)
        else:
            return 0

    @property
    def best(self) -> ThresholdEvaluation:
        """
        Return the `best` threshold evaluation, according to the `recall_best`
        parameter.
        """
        if len(self.threshold_evaluations) > 0:
            if self.recall_best is None:
                best_index = np.argmax(self.F2)
            else:
                best_index = np.argmax(self.recall > self.recall_best)
            return self.threshold_evaluations[best_index]

    def add_threshold_evaluation(self, new: ThresholdEvaluation):
        """
        Inserts the given object into `self.threshold_evaluations`, such that
        `self.thresholds` stays sorted from high to low.
        """
        if len(self.thresholds) == 0:
            insertion_index = 0
        elif new.threshold < np.min(self.thresholds):
            insertion_index = len(self.thresholds)
        else:
            insertion_index = np.argmin(new.threshold < self.thresholds)
        self.threshold_evaluations.insert(insertion_index, new)

    def get_next_threshold(self, threshold_range: Tuple[float, float]) -> float:
        """
        Succesive calls yield a sequence of thresholds within `threshold_range`
        that cover the [0, 1] domains of `recall` and `precision` well. Assumes
        a reasonably smooth PR-curve.
        """
        if len(self.thresholds) == 0:
            return max(threshold_range)
        elif len(self.thresholds) == 1:
            return min(threshold_range)
        else:
            tes = self.threshold_evaluations
            cd = [len(te.correct_detections) for te in tes]
            drs = [len(te.detected_reference_segs) for te in tes]
            cd_gap = np.abs(np.diff(cd))
            drs_gap = np.abs(np.diff(drs))
            index_largest_cd_gap = np.argmax(cd_gap)
            index_largest_drs_gap = np.argmax(drs_gap)
            largest_cd_gap = cd_gap[index_largest_cd_gap]
            largest_drs_gap = drs_gap[index_largest_drs_gap]
            if largest_cd_gap > largest_drs_gap:
                index = index_largest_cd_gap
            else:
                index = index_largest_drs_gap
            surrounding_thresholds = self.thresholds[index : index + 2]
            return np.mean(surrounding_thresholds)
