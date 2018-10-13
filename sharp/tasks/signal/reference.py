"""
Offline detection of sharp wave-ripple segments, for use as a reference to
compare online detections against.
"""
from typing import Tuple

import numpy as np
from luigi import FloatParameter, TupleParameter

from fklab.segments import Segment
from fklab.signals.core import detect_mountains
from fklab.signals.filter import compute_envelope
from sharp.data.types.signal import Signal

from sharp.data.files.numpy import SegmentsFile
from sharp.tasks.base import SharpTask
from sharp.data.files.config import output_root
from sharp.tasks.signal.downsample import DownsampleRecording
from sharp.util import cached, ignore


class MakeReference(SharpTask):

    band: Tuple[float, float] = TupleParameter((100, 250))
    mult_detect: float = FloatParameter(13)
    mult_support: float = FloatParameter(7)
    min_duration: float = FloatParameter(25E-3)
    min_separation: float = FloatParameter(10E-3)

    def requires(self):
        return DownsampleRecording()

    def output(self):
        return SegmentsFile(output_root, "autoref-segments")

    def run(self):
        self.output().write(self.segments)

    @property
    def input_signal(self) -> Signal:
        return self.input().read()

    @property
    @cached
    def envelope(self) -> Signal:
        """ Offline and ripple-only algorithm. """
        # FutureWarning will be fixed in SciPy 1.2, due nov 9 2018
        with ignore(FutureWarning):
            envelope = compute_envelope(
                self.input_signal,
                self.band,
                fs=self.input_signal.fs,
                filter_options=dict(transition_width="10%", attenuation=40),
                smooth_options=dict(kernel="gaussian", bandwidth=7.5E-3),
            )
        return Signal(envelope, self.input_signal.fs)

    @property
    def envelope_center(self):
        return np.median(self.envelope)

    @property
    def envelope_spread(self):
        return np.percentile(self.envelope, 75) - self.envelope_center

    @property
    def threshold_low(self):
        return self.envelope_center + self.mult_support * self.envelope_spread

    @property
    def threshold_high(self):
        return self.envelope_center + self.mult_detect * self.envelope_spread

    @property
    def segments(self) -> Segment:
        """ Start and end times of sharp wave-ripple events. """
        return detect_mountains(
            self.envelope,
            self.envelope.time,
            low=self.threshold_low,
            high=self.threshold_high,
            minimum_duration=self.min_duration,
            allowable_gap=self.min_separation,
        )
