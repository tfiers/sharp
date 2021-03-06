from dataclasses import dataclass
from typing import Optional

from numpy import median, ndarray, percentile

from fklab.segments import Segment


@dataclass
class ThresholdEvaluation:
    """
    Inputs are detections (already classified into {correct, incorrect}),
    reference segments (already classified into {detected, not_detected}),
    and first detection times.

    Calculates detection delays and summarising performance measures based on
    these classifications.
    """

    #
    # ------
    # Inputs
    # ------

    threshold: float
    # (For vectorization in ThresholdSweep).

    correct_detections: ndarray
    # Detection events that hit a reference segment.

    incorrect_detections: ndarray
    # False positive detections.

    first_detections: ndarray
    # For each detected reference segment, the time of the first `detection`
    # that caught it.

    detected_reference_segs: Segment

    undetected_reference_segs: Segment

    detections: Optional[ndarray] = None
    detection_is_correct: Optional[ndarray] = None
    reference_seg_is_detected: Optional[ndarray] = None
    # Optional raw data, for use in the mini-paper notebook
    # (to split detected segments by type: SW-only, ripple-only, SWR-complex)

    #
    # ------
    # Recall
    # ------

    @property
    def num_detected(self):
        return len(self.detected_reference_segs)

    @property
    def num_undetected(self):
        return len(self.undetected_reference_segs)

    @property
    def recall(self) -> float:
        total_refsegs = self.num_detected + self.num_undetected
        if total_refsegs > 0:
            return self.num_detected / total_refsegs
        else:
            return 0

    #
    # ---------
    # Precision
    # ---------

    @property
    def num_correct(self):
        return len(self.correct_detections)

    @property
    def num_incorrect(self):
        return len(self.incorrect_detections)

    @property
    def precision(self) -> float:
        return self.num_correct / (self.num_correct + self.num_incorrect)

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

    @property
    def F3(self):
        return self.get_F_score(3)

    @property
    def F4(self):
        return self.get_F_score(4)

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

    @property
    def abs_delays_median(self):
        return median(self.abs_delays)

    @property
    def abs_delays_Q1(self):
        return percentile(self.abs_delays, 25)

    @property
    def abs_delays_Q3(self):
        return percentile(self.abs_delays, 75)

    @property
    def rel_delays_median(self):
        return median(self.rel_delays)

    @property
    def rel_delays_Q1(self):
        return percentile(self.rel_delays, 25)

    @property
    def rel_delays_Q3(self):
        return percentile(self.rel_delays, 75)
