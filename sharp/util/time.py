import numpy as np


def format_duration(
    duration_in_seconds: float,
    fixed_len: bool = False,
    precision: int = 1,
    auto_ms: bool = True,
) -> str:
    """
    :return:  The given length of time in human-readable format.
    :param duration_in_seconds:  .
    :param fixed_len:  Whether to output leading 0 parts. See examples below.
    :param precision:  Number of digits after the decimal dot. Must be ≥ 0.
    :param auto_ms:  Whether to use "MM.M ms" format when duration is less than
            a second. Ignored when fixed_len is True.

    Output examples for fixed_len = True:
        "1h08m 05.6s"
        "0h08m 05.6s"
        "0h00m 05.6s"

    Output examples for fixed_len = False :
        "1h08m 05.6s"
        "8m 05.6s"
        "5.6s"
        "21.5 ms"
    """
    if not fixed_len and auto_ms and duration_in_seconds < 1:
        return f"{duration_in_seconds * 1000:.{precision}f} ms"
    else:
        total_seconds, fractional_part = divmod(duration_in_seconds, 1)
        # Explicit conversion to int, from float or np.float64 e.g.
        # (int is needed for "02d" format string).
        total_seconds = int(total_seconds)
        total_minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(total_minutes, 60)
        decimal_digits = round(fractional_part * 10 ** precision)
        if precision == 0:
            suffix = "s"
        else:
            suffix = f".{decimal_digits:0{precision}d}s"
        if fixed_len or hours > 0:
            return f"{hours}h{minutes:02d}m {seconds:02d}{suffix}"
        elif minutes > 0:
            return f"{minutes}m {seconds:02d}{suffix}"
        else:
            return f"{seconds}{suffix}"


def view(time, *args) -> np.ndarray:
    """
    A range around some timestamp(s). Useful for plotting.

    :param time:  Number/array-like
    :param args:  either not given, or one of {dt, (t1, t2)}
    :return:  Vector of length 2 when input `t` is scalar.
              N x 2 array when input `t` is a vector of length N.

    Usage:
        view(t)          -->  (t-0.1, t+0.1)
        view(t, dt)      -->  (t-dt, t+dt)
        view(t, t1, t2)  -->  (t-t1, t+t2)
    """
    if len(args) == 0:
        t_before = t_after = 0.1
    elif len(args) == 1:
        t_before, = args
        t_after = t_before
    else:
        t_before, t_after = args

    around = np.array([-t_before, +t_after])

    if hasattr(time, "__iter__"):
        return np.array(time)[:, np.newaxis] + around[np.newaxis, :]
    else:
        return time + around
