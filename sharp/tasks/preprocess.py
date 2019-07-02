from time import time
from typing import Tuple
from warnings import warn

from fklab.signals.multirate import decimate_chunkwise
from sharp.config.config import SharpConfig
from sharp.data.files.base import ArrayFile, ArrayListFile
from sharp.data.files.raw import RawRecordingFile
from sharp.data.files.segments import MultiChannelSegmentsFile
from sharp.data.files.signal import SignalFile
from sharp.data.types.signal import Signal


def downsample_raw(
    input: RawRecordingFile, output: SignalFile, config: SharpConfig
):
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
    output.write(Signal(signal_down, fs_new, "μV"))


def select_channel(
    input: Tuple[MultiChannelSegmentsFile, ArrayListFile], output: ArrayFile
):
    seg_list = input[0].read()
    seg_height_list = input[1].read()
    ...


def trim_recording(
    input: Tuple[SignalFile, MultiChannelSegmentsFile], output: SignalFile
):
    ...
