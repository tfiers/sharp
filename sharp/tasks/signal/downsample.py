from logging import getLogger

from fklab.signals.multirate import decimate_chunkwise
from sharp.config.load import shared_output_root, config
from sharp.data.files.numpy import SignalFile
from sharp.data.types.signal import Signal
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.raw import (
    SingleRecordingFileTask,
    RawRecording_ExistenceCheck,
)

log = getLogger(__name__)


class DownsampleFile(SingleRecordingFileTask):
    output_dir = shared_output_root / "downsampled-recordings"

    def requires(self):
        return RawRecording_ExistenceCheck(file_ID=self.file_ID)

    def output(self):
        return SignalFile(self.output_dir, self.file_ID.ID)

    def work(self):
        raw_recording = self.requires().output()
        fs_orig = raw_recording.fs
        q, remainder = divmod(fs_orig, config.fs_target)
        if remainder > 0:
            log.warning(f"")
        decimate_chunkwise(raw_recording.signal, factor=q)


class GatherDownsampledFiles(SharpTask):
    """
    Read in the Neuralynx continuous recording files in the input directory,
    downsample them, and save the downsampled signals as NumPy arrays.
    """

    output_dir = shared_output_root / "downsampled-recordings"

    def requires(self):
        ...
        # return self.raw_data_dir

    def output(self):
        ...

    #
    # def work(self):
    #     for in_file, out_file in zip(self.raw_data_dir.output(), self.output()):
    #         log.info(f"Downsampling {in_file.name}")
    #         downsampled_data, fs = decimate_chunkwise(
    #             str(in_file), self.fs_target
    #         )
    #         out_file.write(Signal(downsampled_data, fs))

    # def get_multichannel(self):
    #     channels = [file.read() for file in self.output()]
    #     return Signal.from_channels(channels)
    #
    # def get_reference_channel(self) -> Signal:
    #     for file in self.output():
    #         if file.stem == config.reference_channel:
    #             return file.read()
    #     else:
    #         raise ValueError(
    #             f"Cannot find recording channel {config.reference_channel}"
    #         )
