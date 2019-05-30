from abc import ABC
from logging import getLogger
from typing import Tuple

from numpy import abs, array, ceil, log, min
from scipy.signal import hilbert as analytical

from fklab.signals.filter import apply_filter
from fklab.signals.smooth import smooth1d
from sharp.config.load import shared_output_root
from sharp.data.files.numpy import SignalFile
from sharp.data.types.signal import Signal
from sharp.tasks.signal.downsample import DownsampleRawRecording
from sharp.tasks.signal.raw import SingleRecordingFileTask


logr = getLogger(__name__)


output_root = shared_output_root / "offline-filter-envelope"


class CalcEnvelopeFromRawSignal(SingleRecordingFileTask, ABC):
    def requires(self):
        return DownsampleRawRecording(file_ID=self.file_ID)

    def output(self):
        return SignalFile(output_root / self.subdir, self.file_ID.short_str)

    freq_band: Tuple[float, float] = ...  # Hz

    def work(self):
        sig_in = self.requires().output().read()
        logr.info("Read raw signal")
        # We cannot directly use fklab's "compute_envelope", as this function
        # averages all channel envelopes into one.
        bpf_out = apply_filter(
            sig_in,
            axis=0,
            band=self.freq_band,
            fs=sig_in.fs,
            transition_width="20%",
            attenuation=30,
        )
        logr.info("Applied bandpass filter")
        # Use padding to nearest power of 2 or 3 when calculating Hilbert
        # transform for great speedup (via FFT).
        N_orig = sig_in.shape[0]
        N = int(min(array([2, 3]) ** ceil(log(N_orig) / log([2, 3]))))
        envelope_raw_padded = abs(analytical(bpf_out, N=N, axis=0))
        del bpf_out
        envelope_raw = envelope_raw_padded[:N_orig, :]
        logr.info("Calculated raw envelope")
        del envelope_raw_padded
        envelope_smooth = smooth1d(
            envelope_raw, delta=1 / sig_in.fs, kernel="gaussian", bandwidth=4e-3
        )
        logr.info("Smoothed envelope")
        del envelope_raw
        sig_out = Signal(envelope_smooth, sig_in.fs, sig_in.units)
        del envelope_smooth
        self.output().write(sig_out)
        logr.info("Wrote envelope to disk")
        del sig_in, sig_out


class CalcRippleEnvelope(CalcEnvelopeFromRawSignal):
    subdir = "ripple"
    freq_band = (100, 250)


class CalcSharpWaveEnvelope(CalcEnvelopeFromRawSignal):
    subdir = "sharpwave"

    @property
    def freq_band(self):
        shortest_SPW = 20e-3
        highest_f = 1 / (2 * shortest_SPW)
        longest_SPW = 100e-3
        lowest_f = 1 / (2 * longest_SPW)
        return (lowest_f, highest_f)
