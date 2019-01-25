from logging import getLogger

from numpy import empty, int32, ndarray, array

from fklab.segments import Segment
from sharp.config.load import config
from sharp.data.types.evaluation.threshold import ThresholdEvaluation
from sharp.data.types.intersection import SegmentEventIntersection
from sharp.data.types.signal import Signal
from sharp.tasks.signal.util import time_to_index
from sharp.util.misc import compiled

log = getLogger(__name__)


def evaluate_threshold(
    envelope: Signal,
    threshold: float,
    lockout_time: float,
    reference_segs: Segment,
) -> ThresholdEvaluation:
    """
    Evaluate the output of a detector for some threshold and lockout time.
    
    Calculates detections. These are the events where `envelope` crosses
    `threshold` (with a minimum distance of `lockout_time` between detections).
    Classifies detections into {correct, incorrect}, and reference segments
    into {detected, not_detected}. For each detected segment, finds the first
    event that intersected with it.

    :param envelope:
    :param threshold:
    :param lockout_time: Duration after a detection during which no other
                detections can be made. In seconds.
    :param reference_segs:  The (start, stop) tuples that indicate baseline
                "true" SWR segments. In seconds.
    :return:  Initalized ThresholdEvaluation object.
    """
    log.info(f"Evaluating threshold {threshold:.3g}")
    eval_segs = Segment(reference_segs._data - [config.eval_start_extension, 0])
    detections = calc_detections(envelope, threshold, lockout_time)
    intersection = SegmentEventIntersection(eval_segs, detections)
    detection_is_correct = intersection.event_is_in_seg
    reference_seg_is_detected = intersection.num_events_in_seg > 0
    if intersection.first_event_in_seg.size > 0:
        first_detections = detections[intersection.first_event_in_seg]
    else:
        first_detections = array([])
    return ThresholdEvaluation(
        threshold=threshold,
        first_detections=first_detections,
        correct_detections=detections[detection_is_correct],
        incorrect_detections=detections[~detection_is_correct],
        detected_reference_segs=reference_segs[reference_seg_is_detected],
        undetected_reference_segs=reference_segs[~reference_seg_is_detected],
        detections=detections,
        detection_is_correct=detection_is_correct,
        reference_seg_is_detected=reference_seg_is_detected,
    )


def calc_detections(
    envelope: Signal, threshold: float, lockout_time: float
) -> ndarray:
    """
    :param envelope:
    :param threshold:
    :param lockout_time: In seconds
    :return: Array of detection times, in seconds.
    """
    lockout_samples = time_to_index(lockout_time, envelope.fs)
    detection_ix = calc_detection_indices(
        envelope.astype(float), float(threshold), lockout_samples.astype(int)
    )
    detections = detection_ix / envelope.fs
    return detections


@compiled
def calc_detection_indices(
    signal: ndarray, threshold: float, lockout_samples: int
) -> ndarray:
    N = signal.size
    max_detections = N // lockout_samples
    detection_indices = empty(max_detections, dtype=int32)
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
