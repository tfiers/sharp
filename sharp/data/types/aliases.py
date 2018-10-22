"""
Aliases of common types, for readability & consistency of type-hints.
"""

from typing import Sequence, Tuple, Union

import numpy
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


TorchArray = torch.Tensor
NeuralModel = torch.nn.Module

IndexList = numpy.ndarray
EventList = numpy.ndarray  # A 1d array of event times, in seconds.
BooleanArray = numpy.ndarray
Scalar = Union[complex, float, int, bool]
ArrayLike = Union[numpy.ndarray, list, tuple, Scalar]


def subplots(*args, **kwargs) -> Tuple[Figure, Union[Axes, Sequence[Axes]]]:
    return plt.subplots(*args, **kwargs)
