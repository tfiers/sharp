from typing import Sequence, Tuple

import numpy as np

from sharp.datatypes.evaluation.matrix import PerformanceMatrix
from sharp.datatypes.evaluation.sweep import ThresholdSweep
from sharp.datatypes.evaluation.threshold import ThresholdEvaluation
from sharp.datatypes.segments import SegmentArray
from sharp.datatypes.signal import Signal
from sharp.init import sharp_workflow, config
from sharp.util.alias import compiled


@sharp_workflow.task
def distill_performance_matrix(
    mult_detect_matrix: Sequence[Tuple[float, float]],
    ORF_performances: Sequence[ThresholdSweep],
    RNN_performances: Sequence[ThresholdSweep],
) -> PerformanceMatrix:
    ...


@sharp_workflow.task
def score_online_envelope(
    envelope: Signal, reference_SWR_segs: SegmentArray
) -> ThresholdSweep:
    sweep = ThresholdSweep()
    while len(sweep.thresholds) < config.evaluation_config.num_thresholds:
        threshold = sweep.get_next_threshold(envelope.range)
        new_threshold_evaluation = evaluate_threshold(
            envelope, threshold, reference_SWR_segs
        )
        sweep.add_threshold_evaluation(new_threshold_evaluation)

    return sweep


def evaluate_threshold(
    envelope: Signal, threshold: float, reference_segs: SegmentArray
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
    :param reference_segs:  The (start, stop) tuples that indicate baseline
                "true" SWR segments. In seconds.
    :return:  Initalized ThresholdEvaluation object.
    """
    print(f"Evaluating threshold {threshold:.3g}")
    detections = calc_detections(envelope, threshold)
    intersection = reference_segs.contains(detections)
    detection_is_correct = intersection.event_is_in_seg
    reference_seg_is_detected = intersection.num_events_in_seg > 0
    first_detections = detections[intersection.index_of_first_event_in_seg]
    return ThresholdEvaluation(
        threshold=threshold,
        first_detections=first_detections,
        correct_detections=detections[detection_is_correct],
        incorrect_detections=detections[~detection_is_correct],
        detected_reference_segs=reference_segs[reference_seg_is_detected],
        undetected_reference_segs=reference_segs[~reference_seg_is_detected],
    )


def calc_detections(envelope: Signal, threshold: float) -> np.ndarray:
    """
    :param envelope:
    :param threshold:
    :return: Array of detection times, in seconds.
    """
    lockout_samples = round(config.evaluation_config.lockout_time * envelope.fs)
    detection_ix = calc_detection_indices(
        envelope.astype(float), float(threshold), int(lockout_samples)
    )
    detections = detection_ix / envelope.fs
    return detections


@compiled
def calc_detection_indices(
    signal: np.ndarray, threshold: float, lockout_samples: int
) -> np.ndarray:
    N = signal.size
    max_detections = N // lockout_samples
    detection_indices = np.empty(max_detections, dtype=np.int32)
    sample_ix = 0
    det_ix = 0
    while sample_ix < N:
        if signal[sample_ix] >= threshold:
            detection_indices[det_ix] = sample_ix
            det_ix += 1
            sample_ix += lockout_samples + 1
        else:
            sample_ix += 1

    return detection_indices[:det_ix]
