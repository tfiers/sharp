"""
Offline detection of sharp wave-ripple segments, for use as a reference to
compare online detections against.
"""
from typing import Tuple

import numpy as np
from fklab.segments import Segment
from fklab.signals.core import detect_mountains
from fklab.signals.filter import compute_envelope
from luigi import FloatParameter, TupleParameter
from sharp.config.load import intermediate_output_dir
from sharp.data.files.numpy import SegmentsFile
from sharp.data.types.signal import Signal
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.downsample import Downsample
from sharp.util.misc import cached, ignore


class MakeReference(SharpTask):

    band: Tuple[float, float] = TupleParameter((100, 250))
    mult_detect: float = FloatParameter(6.2)
    mult_support: float = FloatParameter(3.6)
    min_duration: float = FloatParameter(25e-3)
    min_separation: float = FloatParameter(10e-3)

    filter_options = dict(transition_width="10%", attenuation=40)
    smooth_options = dict(kernel="gaussian", bandwidth=7.5e-3)

    downsampler = Downsample()

    def requires(self):
        return self.downsampler

    def output(self):
        return SegmentsFile(intermediate_output_dir, "autoref-segments")

    @property
    @cached
    def _input_channel(self) -> Signal:
        return self.downsampler.get_reference_channel()

    def work(self):
        segs = self.calc_segments()
        self.output().write(segs)

    @property
    @cached
    def envelope(self) -> Signal:
        """ Offline and ripple-only algorithm. """
        # FutureWarning will be fixed in SciPy 1.2, due nov 9 2018
        with ignore(FutureWarning):
            envelope = compute_envelope(
                self._input_channel,
                self.band,
                fs=self._input_channel.fs,
                filter_options=self.filter_options,
                smooth_options=self.smooth_options,
            )
        return Signal(envelope, self._input_channel.fs)

    @property
    def envelope_median(self):
        return np.median(self.envelope)

    @property
    def threshold_low(self):
        return self.mult_support * self.envelope_median

    @property
    def threshold_high(self):
        return self.mult_detect * self.envelope_median

    def calc_segments(self) -> Segment:
        """ Start and end times of sharp wave-ripple events. """
        return detect_mountains(
            self.envelope,
            self.envelope.time,
            low=self.threshold_low,
            high=self.threshold_high,
            minimum_duration=self.min_duration,
            allowable_gap=self.min_separation,
        )
