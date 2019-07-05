from time import time
from typing import Sequence
from warnings import warn

from fklab.segments import Segment
from fklab.signals.multirate import decimate_chunkwise
from sharp.config import SharpConfig
from sharp.raw import RawRecordingFile
from sharp.data.types.signal import Signal


def downsample_raw(input: RawRecordingFile, config: SharpConfig) -> Signal:
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


def select_channel(
    mountain_segs: Sequence[Segment],
    mountain_heights: Sequence[Array],
) -> Array:
    ...


def trim_recording(
    full_downsampled_recording: Signal,
    ripple_segs_per_channel: Sequence[Segment],
) -> Signal:
    ...
