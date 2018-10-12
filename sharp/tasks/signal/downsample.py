from warnings import warn

import numpy as np
from luigi import FloatParameter
from scipy.signal import cheby1, filtfilt

from sharp.data.types.aliases import NumpyArray
from sharp.data.types.signal import Signal
from sharp.data.files.numpy import SignalFile
from sharp.data.files.neuralynx import (
    Neuralynx_NCS_Directory,
    Neuralynx_NCS_File,
)
from sharp.tasks.base import SharpTask
from sharp.data.files.config import data_config, output_root
from sharp.tasks.signal.util import time_to_index
from sharp.util import ignore


class DownsampleRecording(SharpTask):
    """
    Read in a Neuralynx continuous recording file, downsample the signal
    present in it, and save the downsampled signal as a NumPy array.
    """

    fs_target = FloatParameter(default=1000)
    # Target sampling rate after downsampling (see `downsample_raw()`). In
    # hertz.

    def requires(self):
        return Neuralynx_NCS_Directory()

    def output(self) -> SignalFile:
        return SignalFile(output_root, filename="downsampled")

    def run(self):
        in_file = self.requires().get_file(
            data_config.probe_number, data_config.electrode_number
        )
        signal = downsample_raw(in_file, self.fs_target)
        self.output().write(signal)


def downsample_raw(
    file: Neuralynx_NCS_File,
    fs_target: float = 1000,
    seconds_per_chunk: float = 10,
    chunk_pad: float = 0.01,
    samples_to_read: int = -1,
) -> Signal:
    """
    Downsamples the given Neuralynx recording.

    Handles recordings that are too large to read and downsample in one go. The
    new sampling frequency is as close to the given `fs_target` as possible
    when using an (integer) subsampling factor.

    :param file:  Path to an *.ncs file.
    :param fs_target:  Target sampling frequency, in Hz.
    :param seconds_per_chunk:
    :param chunk_pad:  To avoid edge effects of the filter at chunk boundaries,
                chunks are extended by this amount of seconds before filtering.
    :param samples_to_read:  If -1 (default), reads and downsamples all recording
                samples.
    :return: downsampled signal.

    Takes between 4 and 40 seconds per 30 minutes of input data at 32 kHz,
    depending on disk read speed. Downsampling is performed by zero-phase
    low-pass filtering (to counter aliasing) and subsequent subsampling.
    """
    if seconds_per_chunk < 2 * chunk_pad:
        warn("chunk_pad should be less than seconds_per_chunk / 2")
    signal = file.signal_mmap
    fs_original = file.fs
    if samples_to_read < 0:
        N = signal.shape[0]
    else:
        N = int(samples_to_read)
    # Create chunk boundaries
    chunk_size = time_to_index(seconds_per_chunk, fs_original)
    chunk_starts = list(range(0, N, chunk_size))
    chunk_ends = chunk_starts[1:] + [N]
    # Downsampling factor
    q = np.round(fs_original / fs_target).astype(int)
    fs_new = fs_original / q
    signal_down = np.empty(N // q)
    b, a = _get_anti_alias_filter(q)
    pad_samples = time_to_index(chunk_pad, fs_original)
    # Iterate over chunks
    for start, end in zip(chunk_starts, chunk_ends):
        # Absolute indices in the signal array
        filter_start = start - pad_samples
        filter_end = end + pad_samples
        # Relative indices in the filtered chunks
        rel_chunk_start = pad_samples
        rel_chunk_end = pad_samples + chunk_size
        # Handle out-of-bounds first and last chunk
        if filter_start < 0:
            filter_start = 0
            rel_chunk_start = 0
            rel_chunk_end = chunk_size
        if filter_end >= N:
            filter_end = end
            rel_chunk_end = pad_samples + end - start
        extended_chunk = signal[slice(filter_start, filter_end)]
        # Anti-alias filter
        # (FutureWarning will be fixed in SciPy 1.2, due nov 9 2018)
        with ignore(FutureWarning):
            extended_chunk_AA = filtfilt(b, a, extended_chunk)
        # Select
        chunk_AA = extended_chunk_AA[rel_chunk_start:rel_chunk_end]
        # Subsample
        chunk_AA_subsampled = chunk_AA[::q]
        bounds = ([start, end] / q).astype(int)
        signal_down[slice(*bounds)] = chunk_AA_subsampled

    return Signal(signal_down, fs_new)


def _get_anti_alias_filter(
    q: int, n: int = 8, rp: float = 0.05, cutoff: float = 0.8
) -> (NumpyArray, NumpyArray):
    """
    (Default arguments correspond to the default anti-aliasing filter of
    scipy.signal.decimate).

    q: Downsampling factor.
    n: Order of the filter.
    rp: Maximum passband ripple, in dB.
    cutoff: Frequency where gain starts dropping below `rp`, relative
        to the Nyquist frequency after downsampling.

    Returns: Filter coefficients b, a.
    """
    return cheby1(n, rp, cutoff / q)
