from logging import getLogger
from typing import Tuple

import numpy as np
from numpy import ndarray
from scipy.signal import cheb2ord, cheby2, lfilter

from sharp.config.load import final_output_dir
from sharp.data.files.stdlib import DictFile
from sharp.data.types.signal import Signal
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.base import EnvelopeMaker

log = getLogger(__name__)


class ApplyOnlineBPF(EnvelopeMaker):

    title = "Single-channel BPF"
    output_filename = "online-BPF"

    def work(self):
        filtered = lfilter(*self.coeffs, self.input_signal)
        envelope = np.abs(filtered)
        envelope_sig = Signal(envelope, self.input_signal.fs)
        self.output().write(envelope_sig)

    @property
    def coeffs(self) -> Tuple[ndarray, ndarray]:
        """ Returns IIR (numer, denom), ie. (b, a) """
        return get_SOTA_online_BPF(self.input_signal.fs)

    @property
    def input_signal(self):
        return self.reference_channel_full.as_vector()


class SaveBPFinfo(SharpTask):

    filtertask = ApplyOnlineBPF()

    def requires(self):
        return self.filtertask

    def output(self):
        return DictFile(final_output_dir, "online-BPF")

    def work(self):
        b, a = self.filtertask.coeffs
        self.output().write(
            {
                "fs": self.filtertask.input_signal.fs,
                "numerator-b": b.tolist(),
                "denominator-a": a.tolist(),
                "order": len(a),
            }
        )


def get_SOTA_online_BPF(
    fs: float,
    left_edge: ndarray = (90, 110),
    right_edge: ndarray = (190, 210),
    passband_ripple: float = 1,
    #   Maximum attenuation in the passband, in dB.,
    attenuation: float = 40,
    #   Minimum attenuation of the stopband, in dB.
) -> (ndarray, ndarray):
    """
    State of the art online IIR band pass filter.

    :param fs: Signal sampling frequency.
    :param left_edge: Left edge of passband. In hertz.
    :param right_edge: idem for right edge.
    :param passband_ripple: Maximum attenuation in the passband, in dB.
    :param attenuation: Minimum attenuation of the stopband, in dB.
    :return: Numerator and denominator coefficients of the IIR filter.

    Notes
    -----
    Design method and parameters are the same as in the file:
     - falcon/tests/filters/iir_ripple_low_delay/matlab_design/iir_ripple_low_delay.filter

    ..from the private Kloostermanlab 'RealTime/falcon' repository.
    """
    f_nyq = fs / 2
    wp = np.array((left_edge[1], right_edge[0])) / f_nyq
    ws = np.array((left_edge[0], right_edge[1])) / f_nyq
    order, critical_freqs = cheb2ord(wp, ws, passband_ripple, attenuation)
    b, a = cheby2(order, attenuation, critical_freqs, "bandpass")
    log.info(f"Online BPF filter order: {order}")
    # len(a) == 2 * order + 1
    return b, a
