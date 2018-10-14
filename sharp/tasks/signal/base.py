from sharp.data.files.config import output_root
from sharp.data.files.numpy import SignalFile
from sharp.data.types.split import TrainTestSplit
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.downsample import DownsampleRecording
from sharp.tasks.signal.reference import MakeReference
from sharp.util import cached


class EnvelopeMaker(SharpTask):

    downsampler = DownsampleRecording()
    reference_maker = MakeReference()

    output_dir = output_root / "output-envelopes"

    def requires(self):
        return (self.downsampler, self.reference_maker)

    def output(self) -> SignalFile:
        """ Implement me """

    @property
    @cached
    def envelope(self):
        return self.output().read()

    @property
    def envelope_train(self):
        return TrainTestSplit(self.envelope).signal_train

    @property
    def envelope_test(self):
        return TrainTestSplit(self.envelope).signal_test

    @property
    def input_signal_all(self):
        return self.downsampler.downsampled_signal

    @property
    def input_signal_train(self):
        return self.downsampler.downsampled_signal_train

    @property
    def reference_segs_train(self):
        return self.reference_maker.reference_segs_train
