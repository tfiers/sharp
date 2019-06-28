from typing import Tuple

from farao import partial
from fklab.signals.filter import apply_filter
from fklab.signals.smooth import smooth1d
from numpy import abs, array, ceil, log, min
from scipy.signal import hilbert as analytical
from sharp.data.files.signal import SignalFile
from sharp.data.types.signal import Signal


RIPPLE_BAND = (100, 250)


def SHARPWAVE_BAND():
    shortest_SPW = 20e-3
    longest_SPW = 100e-3
    highest_f = 1 / (2 * shortest_SPW)
    lowest_f = 1 / (2 * longest_SPW)
    return (lowest_f, highest_f)


# 100__250
# 5__25

def calc_BPF_envelope(
    input: SignalFile, output: SignalFile, freq_band: Tuple[float, float]
):
    """
    :param input:  An unfiltered neural recording.
    :param output:  A smoothed, positive signal of the same shape as "input",
                that is high wherever "input" has high power in the "freq_band".
    :param freq_band:  in Hz.
    """
    print("Reading raw signal")
    sig_in = input.read()
    print("Read raw signal. Applying bandpass filter.")
    # We cannot directly use fklab's "compute_envelope", as this function
    # averages all channel envelopes into one.
    bpf_out = apply_filter(
        sig_in,
        axis=0,
        band=freq_band,
        fs=sig_in.fs,
        transition_width="20%",
        attenuation=30,
    )
    print(
        "Applied bandpass filter. Calculating raw envelope via Hilbert transform."
    )
    # Use padding to nearest power of 2 or 3 when calculating Hilbert
    # transform for great speedup (via FFT).
    N_orig = sig_in.shape[0]
    N = int(min(array([2, 3]) ** ceil(log(N_orig) / log([2, 3]))))
    envelope_raw_padded = abs(analytical(bpf_out, N=N, axis=0))
    del bpf_out
    envelope_raw = envelope_raw_padded[:N_orig, :]
    print("Calculated raw envelope. Smoothing envelope.")
    del envelope_raw_padded
    envelope_smooth = smooth1d(
        envelope_raw, delta=1 / sig_in.fs, kernel="gaussian", bandwidth=4e-3
    )
    print("Smoothed envelope. Writing envelope to disk.")
    sig_out = Signal(envelope_smooth, sig_in.fs, sig_in.units)
    del envelope_smooth
    output.write(sig_out)


def calc_ripple_envelope(input: SignalFile, output: SignalFile, config):
    calc_BPF_envelope(input, output, freq_band=RIPPLE_BAND)


calc_ripple_envelope = partial(
    calc_BPF_envelope, "calc_ripple_envelope", freq_band=RIPPLE_BAND
)
calc_sharpwave_envelope = partial(
    calc_BPF_envelope, "calc_sharpwave_envelope", freq_band=SHARPWAVE_BAND()
)
