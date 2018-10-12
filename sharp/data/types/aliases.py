"""
Aliases of common types, for readability & consistency of type-hints.
"""

from typing import Union

import matplotlib.axes
import matplotlib.figure
import numpy as np
import torch

Figure = matplotlib.figure.Figure
Axes = matplotlib.axes.Axes

NumpyArray = np.ndarray
TorchArray = torch.Tensor
NeuralModel = torch.nn.Module

IndexList = NumpyArray
EventList = NumpyArray  # A 1d array of event times, in seconds.
BooleanArray = NumpyArray
Scalar = Union[complex, float, int, bool]
ArrayLike = Union[NumpyArray, list, tuple, Scalar]
