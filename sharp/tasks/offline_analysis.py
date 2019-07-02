from typing import Tuple, Optional

from numpy import median, array, ceil, log
from scipy.signal import hilbert as analytical

from farao import partial
from fklab.signals.filter import apply_filter
from fklab.signals.smooth import smooth1d
from sharp.data.files.segments import MultiChannelSegmentsFile, SegmentsFile
from sharp.data.files.base import ArrayListFile, ArrayFile
from sharp.data.files.signal import SignalFile
from sharp.data.types.signal import Signal


RIPPLE_BAND = (100, 250)


def SHARPWAVE_BAND():
    shortest_SPW = 20e-3
    longest_SPW = 100e-3
    highest_f = 1 / (2 * shortest_SPW)
    lowest_f = 1 / (2 * longest_SPW)
    return (lowest_f, highest_f)


MIN_SEGMENT_DURATION: float = 25e-3
MIN_SEGMENT_SEPARATION: Optional[float] = None


def calc_BPF_envelope(
    input: SignalFile, output: SignalFile, freq_band: Tuple[float, float]
):
    """
    :param input:  An unfiltered neural recording.
    :param output:  A smoothed, positive signal of the same shape as "input",
                that is high wherever "input" has high power in the "freq_band".
    :param freq_band:  in Hz.
    """
    print("Reading raw signal")
    sig_in = input.read()
    print("Read raw signal. Applying bandpass filter.")
    # We cannot directly use fklab's "compute_envelope", as this function
    # averages all channel envelopes into one.
    bpf_out = apply_filter(
        sig_in,
        axis=0,
        band=freq_band,
        fs=sig_in.fs,
        transition_width="20%",
        attenuation=30,
    )
    print(
        "Applied bandpass filter. Calculating raw envelope via Hilbert transform."
    )
    # Use padding to nearest power of 2 or 3 when calculating Hilbert
    # transform for great speedup (via FFT).
    N_orig = sig_in.shape[0]
    N = int(min(array([2, 3]) ** ceil(log(N_orig) / log([2, 3]))))
    envelope_raw_padded = abs(analytical(bpf_out, N=N, axis=0))
    del bpf_out
    envelope_raw = envelope_raw_padded[:N_orig, :]
    print("Calculated raw envelope. Smoothing envelope.")
    del envelope_raw_padded
    envelope_smooth = smooth1d(
        envelope_raw, delta=1 / sig_in.fs, kernel="gaussian", bandwidth=4e-3
    )
    print("Smoothed envelope. Writing envelope to disk.")
    sig_out = Signal(envelope_smooth, sig_in.fs, sig_in.units)
    del envelope_smooth
    output.write(sig_out)


calc_ripple_envelope = partial(
    calc_BPF_envelope, "calc_ripple_envelope", freq_band=RIPPLE_BAND
)
calc_sharpwave_envelope = partial(
    calc_BPF_envelope, "calc_sharpwave_envelope", freq_band=SHARPWAVE_BAND()
)


def detect_mountains(
    input: SignalFile, output: MultiChannelSegmentsFile, mult_detect: float
):
    """
    :param input:  A signal envelope.
    :param output:  For each channel in "input", the segments where this
            channel crosses a threshold determined by "mult_detect".
    :param mult_detect:  Determines the detection threshold; see def threshold.
    """
    print("Reading in signal")
    envelope = input.read()
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
    output.write(seg_list)
    units = envelope.units
    with output.open_file_for_write() as f:
        f.attrs[f"Support threshold ({units})"] = thr_support
        f.attrs[f"Detect threshold ({units})"] = thr_detect


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
    input: Tuple[SignalFile, MultiChannelSegmentsFile], output: ArrayListFile
):
    envelope = input[0].read()
    seg_list = input[1].read()
    ...


def calc_pairwise_channel_differences(input: SignalFile, output: SignalFile):
    ...


def calc_SWR_segments(
    input: Tuple[SignalFile, ArrayFile, SignalFile, ArrayFile],
    output: SegmentsFile,
    mult_detect_ripple: float,
    mult_detect_SW: float,
):
    ...
