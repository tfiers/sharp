from logging import getLogger
from time import time

from fklab.signals.multirate import decimate_chunkwise
from sharp.config.load import config, shared_output_root
from sharp.data.files.numpy import SignalFile
from sharp.data.types.signal import Signal
from sharp.tasks.base import WrapperTask
from sharp.tasks.signal.raw import (
    RawRecording_ExistenceCheck,
    SingleRecordingFileTask,
)


class DownsampleRawRecording(SingleRecordingFileTask):
    output_dir = shared_output_root / "downsampled-recordings"

    def requires(self):
        return RawRecording_ExistenceCheck(file_ID=self.file_ID)

    def output(self):
        return SignalFile(self.output_dir, self.file_ID.short_str)

    def work(self):
        raw_recording = self.requires().output()
        fs_orig = raw_recording.fs
        q, remainder = divmod(fs_orig, config.fs_target)
        fs_new = fs_orig / q
        if remainder > 0:
            getLogger(__name__).warning(
                f"Original sampling rate of {self.file_ID} ({fs_orig} Hz) is"
                f" not an integer multiple of the target sampling rate"
                f" ({config.fs_target} Hz). Sampling rate after downsampling"
                f" will instead be {fs_new} Hz."
            )
        self.update_status(
            f"Decimating {self.file_ID} ({self.file_ID.path}) of size"
            f" {self.file_ID.path.stat().st_size / 1E9:.1f} GB by a factor {q}."
        )
        t_prev = time()

        def track_downsampling_progress(progress: float):
            global t_prev
            t_now = time()
            time_passed = t_now - t_prev
            if time_passed > 5:
                self.update_progress(progress)
                t_prev = t_now

        signal_down = decimate_chunkwise(
            raw_recording.signal,
            factor=q,
            loop_callback=track_downsampling_progress,
        )
        signal_down *= raw_recording.to_microvolts
        raw_recording.close()
        self.output().write(Signal(signal_down, fs_new, "μV"))


class DownsampleAllRecordings(WrapperTask):
    def requires(self):
        return (
            DownsampleRawRecording(file_ID=rec_file)
            for rec_file in config.raw_data
        )
