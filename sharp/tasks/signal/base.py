from sharp.data.files.config import output_root
from sharp.data.files.numpy import SignalFile
from sharp.data.types.split import TrainTestSplit
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.downsample import DownsampleRecording
from sharp.tasks.signal.reference import MakeReference
from sharp.util import cached


class InputDataMixin:
    _downsampler = DownsampleRecording()
    _reference_maker = MakeReference()

    input_data_makers = (_downsampler, _reference_maker)
    # Should be included in the return values of a Task's `requires()`.

    @property
    def input_signal_all(self):
        return self._downsampler.downsampled_signal

    @property
    def input_signal_train(self):
        return self._downsampler.downsampled_signal_train

    @property
    def input_signal_test(self):
        return self._downsampler.downsampled_signal_test

    @property
    def reference_segs_all(self):
        return self._reference_maker.reference_segs

    @property
    def reference_segs_train(self):
        return self._reference_maker.reference_segs_train

    @property
    def reference_segs_test(self):
        return self._reference_maker.reference_segs_test


class EnvelopeMaker(SharpTask, InputDataMixin):

    output_dir = output_root / "output-envelopes"

    def requires(self):
        return self.input_data_makers

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
