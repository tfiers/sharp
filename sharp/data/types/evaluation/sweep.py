from typing import List, Tuple

from numpy import abs, argmax, argmin, array, diff, mean, min, ndarray, trapz

from sharp.config.load import config
from sharp.data.types.evaluation.threshold import ThresholdEvaluation
from sharp.util.misc import cached


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
    """
    Evaluates an algorithm output envelope, for a range of different detection
    thresholds.
    """

    threshold_evaluations: List[ThresholdEvaluation]
    # Always ordered from highest to lowest threshold.

    def __init__(self):
        self.threshold_evaluations = []

    # FYI: We do not make this a proper dataclass, as `dataclass` does not
    # recognize the following calculated properties, and takes them as init
    # args (which they should not be).
    threshold: ndarray = vectorizing_property("threshold")
    num_detected: ndarray = vectorizing_property("num_detected")
    num_undetected: ndarray = vectorizing_property("num_undetected")
    num_correct: ndarray = vectorizing_property("num_correct")
    num_incorrect: ndarray = vectorizing_property("num_incorrect")
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

    @property
    def AUC(self) -> float:
        """ Area under precision-recall curve. """
        if len(self.threshold_evaluations) > 0:
            return trapz(self.precision, self.recall)
        else:
            return 0

    @property
    @cached
    def max_F1(self):
        return max(self.F1)

    def at_recall(
        self, recall: float = config.selected_recall
    ) -> ThresholdEvaluation:
        if len(self.threshold_evaluations) > 0:
            return self.threshold_evaluations[argmax(self.recall > recall)]

    def at_max_F1(self) -> ThresholdEvaluation:
        if len(self.threshold_evaluations) > 0:
            return self.threshold_evaluations[argmax(self.F1)]

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
            num_correct_gap = abs(diff(self.num_correct))
            num_detected_gap = abs(diff(self.num_detected))
            index_largest_nc_gap = argmax(num_correct_gap)
            index_largest_nd_gap = argmax(num_detected_gap)
            largest_nc_gap = num_correct_gap[index_largest_nc_gap]
            largest_nd_gap = num_detected_gap[index_largest_nd_gap]
            if largest_nc_gap > largest_nd_gap:
                index_gap = index_largest_nc_gap
            else:
                index_gap = index_largest_nd_gap
            thresholds_gap = self.thresholds[index_gap : index_gap + 2]
            return mean(thresholds_gap)
