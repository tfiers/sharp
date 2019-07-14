from time import time
from typing import Sequence, Tuple
from warnings import warn

import numpy as np

from fklab.signals.multirate import decimate_chunkwise
from sharp.datatypes.raw import RawRecordingFile
from sharp.datatypes.segments import SegmentArray
from sharp.datatypes.signal import Signal
from sharp.init import config, sharp_workflow


@sharp_workflow.task
def downsample_raw(input: RawRecordingFile) -> Signal:
    fs_orig = input.fs
    fs_target = config.fs_target
    factor, remainder = divmod(fs_orig, fs_target)
    factor = round(factor)
    fs_new = fs_orig / factor
    if remainder > 0:
        warn(
            f"Original sampling rate of {input} ({fs_orig} Hz) is not an"
            f" integer multiple of the target sampling rate ({fs_target} Hz)."
            f" Sampling rate after downsampling will instead be {fs_new} Hz."
        )
    t_prev = time()

    def track_downsampling_progress(progress: float):
        nonlocal t_prev
        t_now = time()
        time_passed = t_now - t_prev
        if time_passed > 5:
            print(f"Downsampling progress: {progress:.1%}")
            t_prev = t_now

    print(f"Decimating {input} of size {input.size} by a factor {factor}")
    signal_down = decimate_chunkwise(
        input.signal, factor, loop_callback=track_downsampling_progress
    )
    signal_down *= input.to_microvolts
    input.close()
    return Signal(data=signal_down, fs=fs_new, units="μV")


@sharp_workflow.task
def select_channel(
    mountain_segs: [SegmentArray], mountain_heights: Sequence[np.ndarray]
) -> int:
    ...


@sharp_workflow.task
def trim_recording(
    full_downsampled_recording: Signal,
    ripple_segs_per_channel: Sequence[SegmentArray],
) -> Signal:
    ...


@sharp_workflow.task
def split_signal(full: Signal, fraction: float) -> Tuple[Signal, Signal]:
    t_cut = fraction * full.duration
    return (full.time_slice(0, t_cut), full.time_slice(t_cut, full.duration))


@sharp_workflow.task
def split_segmentarray(
    full_segmentarray: SegmentArray, full_sig: Signal, fraction: float
) -> Tuple[SegmentArray, SegmentArray]:
    t_cut = fraction * full_sig.duration
    return (
        full_segmentarray.intersection([0, t_cut]),
        full_segmentarray.intersection([t_cut, full_sig.duration]),
    )
