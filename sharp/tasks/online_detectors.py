from typing import Optional, Tuple, Sequence

from sharp.config.config import SharpConfig
from sharp.data.files.base import ArrayFile
from sharp.data.files.neuralnet import RNNFile
from sharp.data.files.segments import SegmentsFile
from sharp.data.files.signal import SignalFile


def calc_SOTA_output_envelope(
    input: Tuple[SignalFile, ArrayFile], output: SignalFile, config: SharpConfig
):
    ...


def calc_RNN_output_envelope(
    input: Tuple[SignalFile, RNNFile], output: SignalFile, config: SharpConfig
):
    ...


def train_RNN_one_epoch(
    input: Tuple[Optional[RNNFile], SignalFile, SegmentsFile],
    output: RNNFile,
    config: SharpConfig,
):
    ...


def calc_RNN_validation_performance(
    input: Tuple[RNNFile, SignalFile, SegmentsFile],
    output: ArrayFile,
    config: SharpConfig,
):
    ...


def select_RNN(
    input: Tuple[Sequence[RNNFile], Sequence[ArrayFile]], output: RNNFile
):
    ...
