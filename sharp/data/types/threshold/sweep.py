from typing import List, Optional, Tuple

from numpy import abs, argmax, argmin, diff, mean, min, trapz
from numpy.core.multiarray import array, ndarray

from sharp.data.types.threshold.evaluation import ThresholdEvaluation


def vectorizing_property(name: str):
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
        return array(values)

    return property(vectorize_attribute)


class ThresholdSweep:

    threshold_evaluations: List[ThresholdEvaluation]
    # Always ordered from highest to lowest threshold.

    recall_best: Optional[float] = None
    # At which approximate recall value the `best` threshold should be chosen.
    # If not specified or `None`, chooses the threshold with maximal F1-score.

    threshold: ndarray = vectorizing_property("threshold")
    recall: ndarray = vectorizing_property("recall")
    precision: ndarray = vectorizing_property("precision")
    FDR: ndarray = vectorizing_property("FDR")
    F1: ndarray = vectorizing_property("F1")
    F2: ndarray = vectorizing_property("F2")
    abs_delays_median: ndarray = vectorizing_property("abs_delays_median")
    rel_delays_median: ndarray = vectorizing_property("rel_delays_median")
    rel_delays_Q1: ndarray = vectorizing_property("rel_delays_Q1")
    rel_delays_Q3: ndarray = vectorizing_property("rel_delays_Q3")

    thresholds: ndarray = threshold
    # Alias, for readability.

    def __init__(self):
        self.threshold_evaluations = []

    @property
    def AUC(self) -> float:
        """ Area under precision-recall curve. """
        if len(self.threshold_evaluations) > 0:
            return trapz(self.precision, self.recall)
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
                best_index = argmax(self.F1)
            else:
                best_index = argmax(self.recall > self.recall_best)
            return self.threshold_evaluations[best_index]

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
            insertion_index = argmin(new.threshold < self.thresholds)
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
            cd_gap = abs(diff(cd))
            drs_gap = abs(diff(drs))
            index_largest_cd_gap = argmax(cd_gap)
            index_largest_drs_gap = argmax(drs_gap)
            largest_cd_gap = cd_gap[index_largest_cd_gap]
            largest_drs_gap = drs_gap[index_largest_drs_gap]
            if largest_cd_gap > largest_drs_gap:
                index = index_largest_cd_gap
            else:
                index = index_largest_drs_gap
            surrounding_thresholds = self.thresholds[index : index + 2]
            return mean(surrounding_thresholds)
