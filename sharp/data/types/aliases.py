"""
Aliases of common types, for readability & consistency of type-hints.
"""

from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

TorchArray = torch.Tensor
NeuralModel = torch.nn.Module

Scalar = Union[complex, float, int, bool]
ArrayLike = Union[numpy.ndarray, list, tuple, Scalar]


def subplots(*args, **kwargs) -> Tuple[Figure, Union[Axes, Sequence[Axes]]]:
    return plt.subplots(*args, **kwargs)


subplots.__doc__ = plt.subplots.__doc__
