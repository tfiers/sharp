from logging import getLogger
from typing import Sequence

from luigi import FloatParameter

from fklab.signals.multirate import downsample
from sharp.config.load import config, intermediate_output_dir
from sharp.data.files.neuralynx import Neuralynx_NCS_Directory
from sharp.data.files.numpy import SignalFile
from sharp.data.types.signal import Signal
from sharp.tasks.base import SharpTask

log = getLogger(__name__)


class Downsample(SharpTask):
    """
    Read in the Neuralynx continuous recording files in the input directory,
    downsample them, and save the downsampled signals as NumPy arrays.
    """

    raw_data_dir = Neuralynx_NCS_Directory()
    output_dir = intermediate_output_dir / "downsampled-recording"

    fs_target = FloatParameter(default=1000)
    # Target sampling rate after downsampling (see `fklab..downsample()`). In
    # hertz.

    def requires(self):
        return self.raw_data_dir

    def output(self) -> Sequence[SignalFile]:
        return [
            SignalFile(self.output_dir, file.stem)
            for file in self.raw_data_dir.output()
        ]

    def work(self):
        for in_file, out_file in zip(self.raw_data_dir.output(), self.output()):
            log.info(f"Downsampling {in_file.name}")
            downsampled_data, fs = downsample(str(in_file), self.fs_target)
            out_file.write(Signal(downsampled_data, fs))

    def get_multichannel(self):
        channels = [file.read() for file in self.output()]
        return Signal.from_channels(channels)

    def get_reference_channel(self) -> Signal:
        for file in self.output():
            if file.stem == config.reference_channel:
                return file.read()
        else:
            raise ValueError(
                f"Cannot find recording channel {config.reference_channel}"
            )
