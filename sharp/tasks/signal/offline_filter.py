from abc import ABC
from typing import Tuple

from fklab.signals.filter import compute_envelope
from sharp.config.load import shared_output_root
from sharp.data.files.numpy import SignalFile
from sharp.data.types.signal import Signal
from sharp.tasks.signal.downsample import DownsampleRawRecording
from sharp.tasks.signal.raw import SingleRecordingFileTask


output_root = shared_output_root / "offline"


class CalcEnvelopeFromRawSignal(SingleRecordingFileTask, ABC):
    def requires(self):
        return DownsampleRawRecording(file_ID=self.file_ID)

    def output(self):
        return SignalFile(output_root / self.subdir, self.file_ID.short_str)

    freq_band: Tuple[float, float] = ...  # Hz

    def work(self):

        sig_in = self.requires().output().read()
        envelope = compute_envelope(
            sig_in,
            axis=0,
            freq_band=self.freq_band,
            fs=sig_in.fs,
            filter_options=dict(transition_width="20%", attenuation=30),
            smooth_options=dict(kernel="gaussian", bandwidth=4e-3),
        )
        sig_out = Signal(envelope, sig_in.fs, sig_in.units)
        self.output().write(sig_out)


class CalcRippleEnvelope(CalcEnvelopeFromRawSignal):
    subdir = "ripple-envelope"
    freq_band = (100, 250)


class CalcSharpWaveEnvelope(CalcEnvelopeFromRawSignal):
    subdir = "sharpwave-envelope"

    @property
    def freq_band(self):
        shortest_SPW = 20e-3
        highest_f = 1 / (2 * shortest_SPW)
        longest_SPW = 100e-3
        lowest_f = 1 / (2 * longest_SPW)
        return (lowest_f, highest_f)
