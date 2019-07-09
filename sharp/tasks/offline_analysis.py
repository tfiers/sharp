from typing import List, Optional, Tuple

import numpy as np
from numpy import array, ceil, log, median
from scipy.signal import hilbert as analytical

from fileflow.util import partial
from fklab.signals.filter import apply_filter
from fklab.signals.smooth import smooth1d
from sharp.datatypes.segments import SegmentArray
from sharp.datatypes.signal import Signal
from sharp.main import sharp_workflow


RIPPLE_BAND = (100, 250)


def SHARPWAVE_BAND():
    shortest_SPW = 20e-3
    longest_SPW = 100e-3
    highest_f = 1 / (2 * shortest_SPW)
    lowest_f = 1 / (2 * longest_SPW)
    return (lowest_f, highest_f)


MIN_SEGMENT_DURATION: float = 25e-3
MIN_SEGMENT_SEPARATION: Optional[float] = None


@sharp_workflow.task
def calc_BPF_envelope(LFP: Signal, freq_band: Tuple[float, float]) -> Signal:
    """
    :param LFP:  An unfiltered neural recording.
    :param freq_band:  In Hz.
    :return:   A smoothed, positive signal of the same shape as "input",
               that is high wherever "input" has high power in the "freq_band".
    """
    print("Applying bandpass filter.")
    # We cannot directly use fklab's "compute_envelope", as this function
    # averages all channel envelopes into one.
    bpf_out = apply_filter(
        LFP,
        axis=0,
        band=freq_band,
        fs=LFP.fs,
        transition_width="20%",
        attenuation=30,
    )
    print(
        "Applied bandpass filter. Calculating raw envelope via Hilbert transform."
    )
    # Use padding to nearest power of 2 or 3 when calculating Hilbert
    # transform for great speedup (via FFT).
    N_orig = LFP.shape[0]
    N = int(min(array([2, 3]) ** ceil(log(N_orig) / log([2, 3]))))
    envelope_raw_padded = abs(analytical(bpf_out, N=N, axis=0))
    del bpf_out
    envelope_raw = envelope_raw_padded[:N_orig, :]
    print("Calculated raw envelope. Smoothing envelope.")
    del envelope_raw_padded
    envelope_smooth = smooth1d(
        envelope_raw, delta=1 / LFP.fs, kernel="gaussian", bandwidth=4e-3
    )
    print("Smoothed envelope.")
    # We can keep units, as filter passband has amplification of 1.
    return Signal(envelope_smooth, LFP.fs, LFP.units)


calc_ripple_envelope = partial(
    calc_BPF_envelope, "calc_ripple_envelope", freq_band=RIPPLE_BAND
)
calc_sharpwave_envelope = partial(
    calc_BPF_envelope, "calc_sharpwave_envelope", freq_band=SHARPWAVE_BAND()
)


@sharp_workflow.task
def detect_mountains(envelope: Signal, mult_detect: float) -> List[SegmentArray]:
    """
    :param envelope
    :param mult_detect: Determines the detection threshold; see def threshold.
    :return: For each channel in the input envelope, the segments where this
             channel crosses a threshold determined by "mult_detect".
    """
    print("Calculating thresholds")
    thr_support = threshold(envelope, multiplier=0.3)
    thr_detect = threshold(envelope, multiplier=mult_detect)
    seg_list = []
    num_channels = envelope.num_channels
    for channel in range(num_channels):
        print(f"Detecting mountains in channel {channel} / {num_channels}")
        sig: Signal = envelope[:, channel]
        mountain_segs = detect_mountains(
            sig,
            sig.time,
            low=thr_support,
            high=thr_detect,
            minimum_duration=MIN_SEGMENT_DURATION,
            allowable_gap=MIN_SEGMENT_SEPARATION,
        )
        seg_list.append(mountain_segs)
    units = envelope.units
    print(f"Support threshold: {thr_support:.3g} {units}")
    print(f"Detect threshold: {thr_detect:.3g} {units}")
    return seg_list


def threshold(envelope: Signal, multiplier: float):
    # Median (instead of mean + mult * std) avoids skewing by outliers -- such
    # as extreme signal values caused by movement artifacts. Similarly, imagine
    # a baseline background power level (and thus baseline envelope height)
    # interspersed with relatively infrequent bursts of high power. A median
    # will be insensitive to the amount of bursts (up to a point), while the
    # mean will not.
    #
    # Threshold is calculated with envelope flattened over all channels.
    return multiplier * median(envelope)


def calc_mountain_heights(
    envelope: Signal, seg_list: List[SegmentArray]
) -> List[np.ndarray]:
    ...


def calc_pairwise_channel_differences(LFP: Signal) -> Signal:
    ...


def calc_SWR_segments(
    ripple_envelope: Signal,
    ripple_channel: int,
    sharpwave_envelope: Signal,
    sharpwave_channel: int,
    mult_detect_ripple: float,
    mult_detect_SW: float,
) -> SegmentArray:
    ...
