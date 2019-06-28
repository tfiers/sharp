from typing import Optional

from numpy import median
from sharp.data.files.segments import MultiChannelSegmentsFile
from sharp.data.files.signal import SignalFile
from sharp.data.types.signal import Signal

MIN_DURATION: float = 25e-3
MIN_SEPARATION: Optional[float] = None


def detect_mountains(
    input: SignalFile, output: MultiChannelSegmentsFile, mult_detect: float
):
    """
    :param input:  A signal envelope.
    :param output:  For each channel in "input", the segments where this
            channel crosses a threshold determined by "mult_detect".
    :param mult_detect:  Determines the detection threshold ...
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
            minimum_duration=MIN_DURATION,
            allowable_gap=MIN_SEPARATION,
        )
        seg_list.append(mountain_segs)
    output.write(seg_list)
    units = envelope.units
    with output.open_file_for_write() as f:
        f.attrs[f"Support threshold ({units})"] = thr_support
        f.attrs[f"Detect threshold ({units})"] = thr_detect


def threshold(envelope: Signal, multiplier: float):
    # Median (instead of average) avoids skewing by outliers -- such as
    # extreme signal values caused by movement artifacts. Similarly, imagine a
    # baseline background power level (and thus baseline envelope height)
    # interspersed with relatively infrequent bursts of high power. A median
    # will be insensitive to the amount of bursts (up to a point), while the
    # mean will not.
    #
    # Threshold is calculated with envelope flattened over all channels.
    return multiplier * median(envelope)
