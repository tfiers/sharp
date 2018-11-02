"""
Aliases of common types, for readability & consistency of type-hints.
"""

from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import torch
from numpy import ndarray
from matplotlib.axes import Axes
from matplotlib.figure import Figure


TorchArray = torch.Tensor
NeuralModel = torch.nn.Module

Scalar = Union[complex, float, int, bool]
ArrayLike = Union[ndarray, list, tuple, Scalar]

RGB_tup = Tuple[float, float, float]
RGBA_tup = Tuple[float, float, float, float]
Color = Union[str, RGB_tup, RGBA_tup, ndarray]


def subplots(
    nrows=1,
    ncols=1,
    sharex=False,
    sharey=False,
    squeeze=True,
    subplot_kw=None,
    gridspec_kw=None,
    **fig_kw
) -> Tuple[Figure, Union[Axes, Sequence[Axes]]]:
    return plt.subplots(
        nrows, ncols, sharex, sharey, squeeze, subplot_kw, gridspec_kw, **fig_kw
    )


subplots.__doc__ = plt.subplots.__doc__
