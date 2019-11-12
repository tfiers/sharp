"""
Offline detection of sharp wave-ripple segments, for use as a reference to
compare online detections against.
"""
from itertools import product
from typing import Optional, Tuple

from luigi import FloatParameter
from numpy import logical_or, mean, std
from scipy.signal import butter, filtfilt

from fklab.segments import Segment
from fklab.signals.core import detect_mountains
from fklab.signals.filter import compute_envelope
from sharp.config.load import config, intermediate_output_dir
from sharp.data.files.numpy import MultiChannelSegmentsFile
from sharp.data.types.signal import Signal
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.downsample import DownsampleAllRecordings
from sharp.util.misc import cached


class MakeReference(SharpTask):

    mult_detect_SW: float = FloatParameter(default=3)
    mult_detect_ripple: float = FloatParameter(default=2.2)

    mult_support_SW: float = 0.3
    mult_support_ripple: float = 0.3

    min_duration: float = 25e-3
    min_separation: Optional[float] = None
    min_overlap: float = 0.5

    ripple_band: Tuple[float, float] = (100, 250)
    ripple_filter_options = dict(transition_width="10%", attenuation=40)
    ripple_smooth_options = dict(kernel="gaussian", bandwidth=4e-3)
    SW_cutoff: float = 20  # Hz

    downsampler = DownsampleAllRecordings()

    args = (
        tuple(
            dict(mult_detect_SW=mult_SW, mult_detect_ripple=mult_ripple)
            for mult_SW, mult_ripple in product(
                config.mult_detect_SW, config.mult_detect_ripple
            )
        ),
    )

    def requires(self):
        return self.downsampler

    @property
    def output_filename(self):
        return f"SW {self.mult_detect_SW}, ripple {self.mult_detect_ripple}"

    def output(self):
        directory = intermediate_output_dir / "reference-segments"
        return MultiChannelSegmentsFile(directory, self.output_filename)

    def work(self):
        SWR_segs = self.calc_SWR_segments()
        self.output().write(SWR_segs)

    def calc_SWR_segments(self) -> Segment:
        """
        Find ripples that overlap by more than `min_overlap` with a sharp wave,
        and vice versa. Combines both sets of segments.
        """
        SW_segs = self.calc_SW_segments()
        ripple_segs = self.calc_ripple_segments()
        # Matrices that give, for each ripple-SW combo, the fraction that
        # overlaps:
        _, overlap_SW, overlap_ripple = SW_segs.overlap(ripple_segs)
        # Matrix that gives, for each ripple-SW combo, whether either overlaps
        # by more than `min_overlap`:
        SWR_bool = logical_or(
            overlap_SW > self.min_overlap, overlap_ripple > self.min_overlap
        )
        # For each SW, whether it is part of at least one SWR-complex.
        SW_in_SWR_bool = SWR_bool.sum(axis=1) > 0
        # For each ripple, ..idem.
        ripple_in_SWR_bool = SWR_bool.sum(axis=0) > 0
        # Corresponding segments:
        SW_segs_in_SWR = SW_segs[SW_in_SWR_bool]
        ripple_segs_in_SWR = ripple_segs[ripple_in_SWR_bool]
        # Use the ripple segments to define SWR-complex extent.
        SWR_segs = ripple_segs_in_SWR
        return SWR_segs

    @cached
    def calc_ripple_segments(self) -> Segment:
        return detect_mountains(
            self.ripple_envelope,
            self.ripple_envelope.time,
            low=threshold(self.ripple_envelope, self.mult_support_ripple),
            high=self.threshold_detect_ripple,
            minimum_duration=self.min_duration,
            allowable_gap=self.min_separation,
        )

    @cached
    def calc_SW_segments(self) -> Segment:
        return detect_mountains(
            self.SW_envelope,
            self.SW_envelope.time,
            low=threshold(self.SW_envelope, self.mult_support_SW),
            high=self.threshold_detect_SW,
            minimum_duration=self.min_duration,
            allowable_gap=self.min_separation,
        )

    @property
    @cached
    def ripple_envelope(self) -> Signal:
        """ Offline and ripple-only algorithm. """
        ripple_envelope = compute_envelope(
            self.ripple_channel,
            self.ripple_band,
            fs=self.ripple_channel.fs,
            filter_options=self.ripple_filter_options,
            smooth_options=self.ripple_smooth_options,
        )
        return Signal(ripple_envelope, self.ripple_channel.fs)

    @property
    @cached
    def ripple_channel(self) -> Signal:
        return self.downsampler.get_reference_channel()

    @property
    @cached
    def SW_envelope(self) -> Signal:
        return 0.5 * self.toppyr_envelope - 0.5 * self.sr_envelope

    @property
    def toppyr_envelope(self):
        return self.SW_LPF(self.toppyr_channel)

    @property
    def sr_envelope(self):
        return self.SW_LPF(self.sr_channel)

    def SW_LPF(self, signal: Signal) -> Signal:
        fn = signal.fs / 2
        order = 5
        ba = butter(order, self.SW_cutoff / fn, "low")
        out = filtfilt(*ba, signal)
        return Signal(out, signal.fs)

    @property
    def toppyr_channel(self) -> Signal:
        return self.downsampler.get_multichannel()[:, config.toppyr_channel_ix]

    @property
    def sr_channel(self) -> Signal:
        return self.downsampler.get_multichannel()[:, config.sr_channel_ix]

    @property
    def threshold_detect_SW(self):
        return threshold(self.SW_envelope, self.mult_detect_SW)

    @property
    def threshold_detect_ripple(self):
        return threshold(self.ripple_envelope, self.mult_detect_ripple)


def threshold(signal, zscore):
    return mean(signal) + zscore * std(signal)
