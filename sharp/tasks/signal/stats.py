from abc import ABC
from logging import getLogger
from typing import Optional, Type

from numpy import median

from fklab.signals.core import detect_mountains
from sharp.config.load import shared_output_root
from sharp.data.files.numpy import MultiChannelSegmentsFile
from sharp.data.types.signal import Signal
from sharp.tasks.signal.offline_filter import (
    CalcEnvelopeFromRawSignal,
    CalcRippleEnvelope,
    CalcSharpWaveEnvelope,
)
from sharp.tasks.signal.raw import SingleRecordingFileTask


log = getLogger(__name__)


class DetectMountains(SingleRecordingFileTask, ABC):
    """
    - mult_detect and mult_support are specified as multiples of the envelope
        median.
    - min_duration and min_separation are in seconds.
    """

    parent_task_class: Type[CalcEnvelopeFromRawSignal]
    subdir: str

    mult_detect: float = 3
    mult_support: float = 2
    min_duration: float = 25e-3
    min_separation: Optional[float] = None

    def requires(self):
        return self.parent_task_class(file_ID=self.file_ID)

    def output(self):
        return MultiChannelSegmentsFile(
            directory=shared_output_root / "mountains" / self.subdir,
            filename=self.file_ID.short_str,
        )

    @property
    def envelope(self):
        return self.requires().output().read()

    def work(self):
        log.info("Calculating thresholds")
        # Calculate thresholds flattened over all channels
        thr_support = self.threshold(self.mult_support)
        thr_detect = self.threshold(self.mult_detect)
        seg_list = []
        for channel in range(self.envelope.num_channels):
            log.info(f"Detecting mountains in channel {channel}")
            sig: Signal = self.envelope[:, channel]
            mountain_segs = detect_mountains(
                sig,
                sig.time,
                low=thr_support,
                high=thr_detect,
                minimum_duration=self.min_duration,
                allowable_gap=self.min_separation,
            )
            seg_list.append(mountain_segs)
        self.output().write(seg_list)
        with self.output().open_file_for_write() as f:
            f.attrs["Support threshold (uV)"] = thr_support
            f.attrs["Detect threshold (uV)"] = thr_detect

    def threshold(self, multiplier: float):
        # Median avoids skewing by outliers (such as extreme signal values
        # caused by movement artifacts).
        return multiplier * median(self.envelope)


class DetectRipples(DetectMountains):
    parent_task_class = CalcRippleEnvelope
    subdir = "ripples"


class DetectSharpwaves(DetectMountains):
    parent_task_class = CalcSharpWaveEnvelope
    subdir = "sharpwaves"
    mult_detect = 1.5
    mult_support = 1.2
