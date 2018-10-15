"""
Aliases of common types, for readability & consistency of type-hints.
"""

from typing import Union

import matplotlib.axes
import matplotlib.figure
import numpy
import torch

Figure = matplotlib.figure.Figure
Axes = matplotlib.axes.Axes

TorchArray = torch.Tensor
NeuralModel = torch.nn.Module

IndexList = numpy.ndarray
EventList = numpy.ndarray  # A 1d array of event times, in seconds.
BooleanArray = numpy.ndarray
Scalar = Union[complex, float, int, bool]
ArrayLike = Union[numpy.ndarray, list, tuple, Scalar]
