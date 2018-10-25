from sharp.config.params import intermediate_output_dir
from sharp.data.files.numpy import SignalFile
from sharp.data.types.split import TrainTestSplit
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.downsample import Downsample
from sharp.tasks.signal.reference import MakeReference
from sharp.util import cached


class InputDataMixin:
    _downsampler = Downsample()
    _reference_maker = MakeReference()

    input_data_makers = (_downsampler, _reference_maker)
    # Should be included in the return values of a Task's `requires()`.

    # -------------
    # All channels:

    @property
    @cached
    def multichannel_full(self):
        return self._downsampler.get_multichannel()

    @property
    def multichannel_train(self):
        return TrainTestSplit(self.multichannel_full).signal_train

    @property
    def multichannel_test(self):
        return TrainTestSplit(self.multichannel_full).signal_test

    # -----------------
    # Selected channel:

    @property
    @cached
    def reference_channel_full(self):
        return self._downsampler.get_reference_channel()

    @property
    def reference_channel_train(self):
        return TrainTestSplit(self.reference_channel_full).signal_train

    @property
    def reference_channel_test(self):
        return TrainTestSplit(self.reference_channel_full).signal_test

    # -------------------
    # Reference segments:

    @property
    @cached
    def reference_segs_all(self):
        return self._reference_maker.output().read()

    @property
    def reference_segs_train(self):
        return self._split_refsegs.segments_train

    @property
    def reference_segs_test(self):
        return self._split_refsegs.segments_test

    @property
    def _split_refsegs(self):
        return TrainTestSplit(
            self.reference_channel_full, self.reference_segs_all
        )


class EnvelopeMaker(SharpTask, InputDataMixin):

    output_subdir: str = ""
    output_filename: str = ...
    title: str = ...

    def requires(self):
        return self.input_data_makers

    @property
    def output_dir(self):
        return intermediate_output_dir / "output-envelopes" / self.output_subdir

    def output(self):
        return SignalFile(self.output_dir, self.output_filename)

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
