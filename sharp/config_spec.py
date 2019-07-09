from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Sequence, Tuple

import fileflow
from numpy import linspace
from sharp.datatypes.neuralnet import SharpRNN
from sharp.fklab_data import fklab_probe_recordings


@dataclass
class RNNTrainingConfig:
    
    # The remaining fraction of training data is used to select a model.
    tune_fraction: float = 2 / 3
    
    # Number of passes over the training data. Determines training duration.
    num_epochs: int = 60

    # Per training epoch, the training data is divided into short chunks. This
    # is their length, in seconds. Per chunk, the output-vs-target loss is
    # average, the RNN weights are updated, and training progress is logged.
    chunk_duration: float = 1
    
    # After this many seconds, the hidden state between one training chunk and
    # the next is detached. When not detached, RNN weights are updated based on
    # partial derivatives (pdv's) of matrix operations of previous chunks.
    # I.e.: what influence does this weight *x* minutes ago have on the current
    # hidden state (and thus on the loss for the current chunk)?
    backprop_duration: float = 120


class SharpConfig(fileflow.Config):
    # By default, save output files relative to where the workflow is run from.
    output_root: Path = "output/"

    # A collection of paths to ".raw.kwd", ".dat", and ".moz" files containing
    # raw neural recordings, each identified (for short reference) by an
    # arbitrary ID string.
    raw_data: Collection[Tuple[str, Path]] = fklab_probe_recordings

    # Target sampling frequency after downsampling. In hertz.
    fs_target: float = 1000
    
    # The remaining fraction of data is used for reporting detector performance.
    train_fraction: float = 0.75

    mult_detect_ripple: Sequence[float] = tuple(linspace(1, 3, 0.5))
    mult_detect_SW: Sequence[float] = tuple(linspace(1, 3, 0.5))

    RNN_hyperparams: SharpRNN.Hyperparams = SharpRNN.Hyperparams()
    RNN_training: RNNTrainingConfig = RNNTrainingConfig()
