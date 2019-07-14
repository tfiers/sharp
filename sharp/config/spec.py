from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Sequence, Tuple

import numpy as np

import fileflow
from sharp.config.fklab_data import fklab_probe_recordings
from sharp.datatypes.neuralnet import SharpRNN


@dataclass
class RNNTrainingConfig:

    # Fractio of training set used for tuning model parameters.
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


@dataclass
class EvaluationConfig:
    # Duration after a detection during which no other detections can be made.
    # In seconds.
    lockout_time: float = 60e-3

    num_thresholds: int = 60


@dataclass
class SharpConfig(fileflow.Config):
    # By default, save output files relative to where the workflow is run from.
    output_root: Path = "output/"

    # A collection of paths to ".raw.kwd", ".dat", and ".moz" files containing
    # raw neural recordings, each identified (for short reference) by an
    # arbitrary ID string.
    raw_data: Collection[Tuple[str, Path]] = tuple(fklab_probe_recordings)

    # Target sampling frequency after downsampling. In hertz.
    fs_target: float = 1000

    # Fraction of total recording data used for training/selecting a detector.
    # The remaining fraction of data is used for reporting detector performance.
    train_fraction: float = 0.75

    # Multipliers determining ripple and sharp wave detection thresholds.
    ripple_detect_multipliers: Sequence[float] = tuple(np.arange(1, 3, 0.5))
    sharpwave_detect_multipliers: Sequence[float] = tuple(np.arange(1, 3, 0.5))

    evaluation_config: EvaluationConfig = EvaluationConfig()

    RNN_hyperparams: SharpRNN.Hyperparams = SharpRNN.Hyperparams()
    RNN_training: RNNTrainingConfig = RNNTrainingConfig()

    # In Hz.
    online_ripple_filter_passband = (100, 200)
