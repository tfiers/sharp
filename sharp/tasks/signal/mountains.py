from abc import ABC
from typing import Optional, Type

from numpy import median, max

from fklab.signals.core import detect_mountains
from sharp.config.load import shared_output_root
from sharp.data.files.numpy import (
    MultiChannelDataFile,
    MultiChannelSegmentsFile,
)
from sharp.data.types.signal import Signal
from sharp.tasks.signal.offline_filter import (
    CalcBPFEnvelopeFromRawSignal,
    CalcRippleEnvelope,
    CalcSharpWaveEnvelope,
)
from sharp.tasks.signal.raw import SingleRecordingFileTask
from sharp.util.misc import cached


output_root = shared_output_root / "mountains"


class DetectMountains(SingleRecordingFileTask, ABC):
    """
    - mult_detect and mult_support are specified as multiples of the envelope
        median.
    - min_duration and min_separation are in seconds.
    """

    parent: Type[CalcBPFEnvelopeFromRawSignal]
    subdir: str

    mult_detect: float = 3
    mult_support: float = 2
    min_duration: float = 25e-3
    min_separation: Optional[float] = None

    def requires(self):
        return self.parent(file_ID=self.file_ID)

    def output(self):
        return MultiChannelSegmentsFile(
            directory=output_root / "segments" / self.subdir,
            filename=self.file_ID.short_str,
        )

    @cached
    def threshold(self, multiplier: float):
        # Median (instead of average) avoids skewing by outliers (such as
        # extreme signal values caused by movement artifacts).
        #
        # Threshold is calculated with envelope flattened over all channels.
        return multiplier * median(self.envelope)

    thr_support = property(lambda self: self.threshold(self.mult_support))
    thr_detect = property(lambda self: self.threshold(self.mult_detect))

    @property
    def envelope(self):
        return self.requires().output().read()

    def work(self):
        self.update_status("Calculating thresholds")
        seg_list = []
        num_channels = self.envelope.num_channels
        for channel in range(num_channels):
            self.update_status(f"Detecting mountains in channel {channel}")
            self.update_progress(channel / num_channels)
            sig: Signal = self.envelope[:, channel]
            mountain_segs = detect_mountains(
                sig,
                sig.time,
                low=self.thr_support,
                high=self.thr_detect,
                minimum_duration=self.min_duration,
                allowable_gap=self.min_separation,
            )
            seg_list.append(mountain_segs)
        self.output().write(seg_list)
        with self.output().open_file_for_write() as f:
            f.attrs["Support threshold (uV)"] = self.thr_support
            f.attrs["Detect threshold (uV)"] = self.thr_detect


class DetectRipples(DetectMountains):
    parent = CalcRippleEnvelope
    subdir = "ripples"


class DetectSharpwaves(DetectMountains):
    parent = CalcSharpWaveEnvelope
    subdir = "sharpwaves"
    mult_detect = 1.5
    mult_support = 1.2


class CalcMountainHeights(SingleRecordingFileTask, ABC):
    parent: Type[DetectMountains]
    subdir: str

    def requires(self):
        return self.parent(file_ID=self.file_ID)

    def output(self):
        return MultiChannelDataFile(
            directory=output_root / "heights" / self.subdir,
            filename=self.file_ID.short_str,
        )

    def work(self):
        mountain_detector = self.requires()
        segs_list = mountain_detector.output().read()
        heights_list = []
        for channel, segs in enumerate(segs_list):
            envelope: Signal = mountain_detector.envelope[:, channel]
            heights = []
            for data_in_seg in envelope.extract(segs):
                heights.append(max(data_in_seg))
            heights_list.append(heights)
        self.output().write(heights_list)


class CalcSharpwaveStrength(CalcMountainHeights):
    parent = DetectSharpwaves
    subdir = "sharpwaves"


class CalcRippleStrength(CalcMountainHeights):
    parent = DetectRipples
    subdir = "ripples"
