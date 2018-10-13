from fklab.segments import Segment
from sharp.data.types.signal import Signal
from sharp.data.files.config import output_root

from sharp.data.files.numpy import SignalFile
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

    @property
    @cached
    def input_signal(self) -> Signal:
        """ Input signal, as a one-column matrix. """
        return self.downsampler.output().read().as_matrix()

    @property
    @cached
    def reference_segs(self) -> Segment:
        return self.reference_maker.output().read()

    def output(self) -> SignalFile:
        """ Implement me """
