from typing import Type

from luigi import BoolParameter, FloatParameter, TaskParameter
from sharp.data.types.slice import Slice
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.base import EnvelopeMaker
from sharp.tasks.signal.downsample import DownsampleRecording
from sharp.tasks.signal.reference import MakeReference
from sharp.util import cached


class TrainTestSplitter(SharpTask):
    """
    Cuts up the downsampled input signal, reference segments, and an optional
    output envelope into test and train parts. Does not write files to disk
    (that'd be redundant). Instead recalculates slices on demand, in memory
    only.
    """

    # Useful for choosing split boundary: relative timestamps of Kloosterman
    # Lab scientists labelling L2 data ('labelface'):
    #  - common set last event = 161 / 2040 = 0.0789
    #  - last labeller last event = 860 / 2040 = 0.4216

    train_fraction = FloatParameter(0.9)
    # Border between training and testing data, as a fraction of total signal
    # duration.
    train_first = BoolParameter(False)
    # Whether the training data comes before the test data or not.

    envelope_maker_type: Type[EnvelopeMaker] = TaskParameter(default=None)

    downsampler = DownsampleRecording()
    reference_maker = MakeReference()

    @property
    def envelope_maker(self) -> EnvelopeMaker:
        return self.envelope_maker_type()

    def requires(self):
        dependencies = (self.downsampler, self.reference_maker)
        if self.envelope_maker_type:
            dependencies += (self.envelope_maker,)
        return dependencies

    @property
    def train(self):
        return Slice(
            self._train_slice_fractions,
            self._full_input,
            self._full_envelope,
            self._full_reference_segs,
        )

    @property
    def test(self):
        return Slice(
            self._test_slice_fractions,
            self._full_input,
            self._full_envelope,
            self._full_reference_segs,
        )

    @property
    def _train_slice_fractions(self):
        if self.train_first:
            return (0, self.train_fraction)
        else:
            return (1 - self.train_fraction, 1)

    @property
    def _test_slice_fractions(self):
        if self.train_first:
            return (self.train_fraction, 1)
        else:
            return (0, 1 - self.train_fraction)

    @property
    @cached
    def _full_input(self):
        return self.downsampler.output().read()

    @property
    @cached
    def _full_envelope(self):
        return self.envelope_maker.output().read()

    @property
    @cached
    def _full_reference_segs(self):
        return self.reference_maker.output().read()
