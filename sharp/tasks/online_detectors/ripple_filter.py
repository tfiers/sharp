import numpy as np
from scipy.signal import butter, lfilter, normalize

from sharp.datatypes.signal import Signal
from sharp.init import config, sharp_workflow


@sharp_workflow.task
def calc_online_ripple_filter_envelope(
    LFP: Signal, ripple_channel: int
) -> Signal:
    passband = config.online_ripple_filter_passband
    # Construct the best online ripple filter, as determined in Master's thesis:
    # A combination of a Butterworth (i.e IIR) low and highpass filter.
    b_hi, a_hi = butter(6, passband[0], "high", fs=LFP.fs)
    b_lo, a_lo = butter(1, passband[1], "low", fs=LFP.fs)
    b = np.polymul(b_hi, b_lo)
    a = np.polymul(a_hi, a_lo)
    bpf = normalize(b, a)
    wideband = LFP[:, ripple_channel]
    bandpass_filtered = lfilter(*bpf, wideband)
    online_envelope = np.abs(bandpass_filtered)
    # We can keep units, because filter has amplification 1 in passband.
    return Signal(online_envelope, LFP.fs, LFP.units)
