import functools
from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numba
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# Short alias for caching function outputs in memory. (We don't care that it's
# an _LRU_ cache, and we don't want to change the cache size).
# NOTE: when combined with @property, @cached goes nearest to the function.
cached = functools.lru_cache(maxsize=256)

# Another short alias
compiled = numba.jit(cache=True, nopython=True)


# "plt.subplots" lacks type annotations for its return values (namely the
# created fig and axes). This means no code completion on these objects, which
# is annoying as they are used a lot. Here we make a copy of "plt.subplots",
# adding return type hints.
#
# fmt: off
def subplots(
    nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw
) -> Tuple[Figure, Union[Axes, Sequence[Axes]]]:
    return plt.subplots(nrows, ncols, sharex, sharey, squeeze, subplot_kw, gridspec_kw, **fig_kw)
subplots.__doc__ = plt.subplots.__doc__
# fmt: on
