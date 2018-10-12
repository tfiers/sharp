from typing import Type

from luigi import TaskParameter, TupleParameter

from sharp.data.types.aliases import NumpyArray
from sharp.data.types.signal import Signal
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.base import EnvelopeMaker
from sharp.tasks.signal.util import fraction_to_index
from sharp.util import cached


class EvaluationSliceMaker(SharpTask):
    """
    Cuts out a slice of the input signal, the reference segments, and the
    output envelope of a SWR detector. This slice is used to evaluate the
    detector.
    """

    eval_seg = TupleParameter(
        default=(0, 0.1),
        description=(
            "2-tuple that defines which input data will be used to evaluate "
            "the envelopes. In fractions of total signal duration."
        ),
        significant=False,
    )

    envelope_maker: Type[EnvelopeMaker] = TaskParameter()

    def requires(self) -> EnvelopeMaker:
        return self.envelope_maker()

    @property
    def eval_range(self) -> NumpyArray:
        """ (start, stop) times of the evaluation slice. """
        return self._eval_range_indices / self.input_signal.fs

    @property
    @cached
    def envelope(self) -> Signal:
        return self._envelope_full[slice(*self._eval_range_indices)]

    @property
    @cached
    def reference_segs(self):
        ref_segs_full = self.requires().reference_segs
        return ref_segs_full.intersection(self.eval_range)

    @property
    @cached
    def input_signal(self) -> Signal:
        input_sig_full = self.requires().input_signal
        return input_sig_full[slice(*self._eval_range_indices)]

    @property
    def _eval_range_indices(self) -> NumpyArray:
        return fraction_to_index(self._envelope_full, self.eval_seg)

    @property
    @cached
    def _envelope_full(self) -> Signal:
        return self.requires().output().read()
