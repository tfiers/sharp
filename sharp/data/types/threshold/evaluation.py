from dataclasses import dataclass

from numpy import median, percentile
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

    @property
    def abs_delays_median(self):
        return median(self.abs_delays)

    @property
    def rel_delays_median(self):
        return median(self.rel_delays)

    @property
    def rel_delays_Q1(self):
        return percentile(self.rel_delays, 25)

    @property
    def rel_delays_Q3(self):
        return percentile(self.rel_delays, 75)
