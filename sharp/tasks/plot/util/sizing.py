import matplotlib as mpl
import matplotlib.pyplot as plt


def pixels_to_coords(pixels, trans, orient="both"):
    """
    Converts a length in pixels to a length in data or axes coordinates.

    Parameters
    ----------
    pixels : float
        Length in display coordinates (pixels).
    trans : mpl.transforms.Transform
        Data (ax.transData), axes (ax.transAxes), or blended transform.
    orient : {'h', 'v', 'both'}
        Which dimension to return.

    Returns
    -------
    length : float  or  (float, float)
        Length in `trans` coordinates.
        A tuple when orient="both". A scalar otherwise.
    """
    # Get the origin in display coordinates:
    origin = trans.transform([0, 0])
    # Transform display coordinates to data or axes coordinates:
    x, y = trans.inverted().transform(origin + pixels)
    if orient == "h":
        return x
    elif orient == "v":
        return y
    elif orient == "both":
        return x, y


def get_fontsize():
    """
    Returns
    -------
    pixels : float
        The current Matplotlib font size, in pixels.
    """
    INCHES_PER_POINT = 1 / 72
    pixels_per_inch = plt.gcf().dpi
    points = mpl.rcParams["font.size"]
    inches = points * INCHES_PER_POINT
    pixels = inches * pixels_per_inch
    return pixels
