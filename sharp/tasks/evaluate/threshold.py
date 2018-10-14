from logging import getLogger

import numba
import numpy as np
from sharp.data.types.evaluation import ThresholdEvaluation
from sharp.data.types.intersection import SegmentEventIntersection
from sharp.data.types.signal import Signal
from sharp.data.types.aliases import NumpyArray
from sharp.tasks.signal.util import time_to_index

from fklab.segments import Segment

log = getLogger(__name__)


def evaluate_threshold(
    envelope: Signal,
    threshold: float,
    lockout_time: float,
    reference_segs: Segment,
) -> ThresholdEvaluation:
    """
    Evaluate the output of a detector for a single threshold and lockout time.

    lockout_time: Time after a detection during which no other detections
        can be made. In seconds.

    Detections are the events where `envelope` crosses `threshold` (with a
    minimum distance of `lockout_time` between detections).
    """
    log.info(
        f"Evaluating threshold {threshold:.3g} "
        f"with a lockout time of {1000 * lockout_time:.3g} ms."
    )
    lockout_samples = time_to_index(lockout_time, envelope.fs)
    detection_ix = calc_detection_indices(
        envelope.astype(float), float(threshold), lockout_samples.astype(int)
    )
    detections = detection_ix / envelope.fs
    intersection = SegmentEventIntersection(reference_segs, detections)
    return ThresholdEvaluation(
        detections, reference_segs, intersection, threshold
    )


@numba.jit("i4[:](f8[:], f8, i4)", nopython=True, cache=True)
def calc_detection_indices(
    signal: NumpyArray, threshold: float, lockout_samples: int
) -> NumpyArray:
    N = signal.size
    max_detections = N // lockout_samples
    detection_indices = np.empty(max_detections, dtype=np.int32)
    i = 0  # Sample nr.
    j = 0  # Detection nr.
    while i < N:
        if signal[i] >= threshold:
            detection_indices[j] = i
            j += 1
            i += lockout_samples + 1
        else:
            i += 1

    return detection_indices[:j]
