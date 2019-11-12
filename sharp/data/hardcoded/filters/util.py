from numpy import angle, diff, log10, nan, ndarray, percentile, unwrap, where
from scipy.signal import savgol_filter


def gain(H):
    """
    :param H:  Array of complex frequency responses of a filter.
    :return:  Gain of the filter (dimensionless).
    """
    return abs(H)


def dB(x):
    return 20 * log10(x)


def phase(H):
    """
    :return:  Phase of the filter, in radians.
    """
    return unwrap(angle(H))


def group_delay(H, f):
    """
    :param f:  Equi-spaced frequencies where H is calculated, in Hz.
    :return:  Group delays of the filter, in milliseconds.
    """
    df = diff(f[:2])
    phi = phase(H)
    tau = -savgol_filter(phi, 5, 3, deriv=1, delta=df)  # In seconds (1/Hz)
    remove_outliers(tau)
    return 1000 * tau


def remove_outliers(data: ndarray, m: float = 20):
    mag = abs(data)
    Q1, med, Q3 = percentile(mag, [25, 50, 75])
    threshold = med + m * (Q3 - Q1)
    outlier_ix = where(mag > threshold)
    data[outlier_ix] = nan
