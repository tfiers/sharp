from dataclasses import dataclass
from random import uniform
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn
import torch.optim

from sharp.datatypes.neuralnet import Input_TargetPair, SharpRNN
from sharp.datatypes.segments import SegmentArray
from sharp.datatypes.signal import Signal
from sharp.init import config, sharp_workflow


@sharp_workflow.task
def get_init_RNN(LFP: Signal) -> SharpRNN:
    """ Initialize a fresh model for the first epoch. """
    hyperparams = config.RNN_hyperparams
    hyperparams.num_input_channels = LFP.num_channels
    return SharpRNN(hyperparams)


...
# The cost function quantifies RNN performance by comparing network output with
# target signal (with lower values being better). This function applies a
# sigmoid squashing function to its first argument, and compares the result of
# this squashing with its second argument (a binary signal), by calculating the
# binary cross entropy between them.
signature = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
cost_function: signature = torch.nn.BCEWithLogitsLoss(reduction="sum")


@sharp_workflow.task
def tune_RNN_one_epoch(
    model: SharpRNN, LFP: Signal, reference_SWRs: SegmentArray
) -> SharpRNN:
    chunks = create_chunks(
        LFP,
        reference_SWRs,
        chunk_duration=config.RNN_training.chunk_duration,
        randomize_first_chunk_duration=True,
    )
    # The gradient descent algorithm 'AdaMax' keeps track of exponentially
    # weighted and normalised moving averages of the partial derivatives
    # (pdv's) of the loss function wrt. each weight, and uses these to update
    # the weights.
    weight_updater = torch.optim.Adamax(model.parameters())
    h = model.get_init_h()
    time_since_last_detach = 0
    logger = ChunkProgressLogger(num_chunks=len(chunks))
    for chunk in chunks:
        if time_since_last_detach > config.RNN_training.backprop_duration:
            # Cut the gradient backpropagation link between previous chunks and
            # the current chunk.
            h = h.detach()
            time_since_last_detach = 0
        else:
            time_since_last_detach += chunk.input_sig.duration
        # Run chunk of input signal through RNN.
        output, h = model.forward(chunk.torch_input, h)
        # Evaluate cost function (calculate 'loss') by comparing output with
        # the target signal.
        loss = cost_function(output, chunk.torch_target)
        # Bookkeeping.
        logger.update(chunk, loss)
        # Calculate partial derivative of loss wrt. each model weight. As
        # weights are shared between time steps, these pdv's accumulate for
        # each timestep.
        loss.backward()
        # Perform gradient descent step: update each weight of the model based
        # on the "gradient" value stored alongside it (this "gradient" value is
        # the sum of the pdv's of "loss" wrt to that weight at different time
        # steps).
        weight_updater.step()
        # Empty the partial derivative accumulator of each weight. (The weights
        # have just been updated using the pdv's of the current chunk's loss;
        # we do not want to reuse this information for the next chunk).
        model.zero_grad()

    print("Finished training epoch")


@sharp_workflow.task
def calc_mean_sample_loss(
    model: SharpRNN, LFP: Signal, reference_SWRs: SegmentArray
) -> float:
    _, logger = _apply_RNN_chunked(model, LFP, reference_SWRs)
    return logger.mean_sample_loss


@sharp_workflow.task
def calc_RNN_envelope(
    model: SharpRNN, LFP: Signal, reference_SWRs: SegmentArray
) -> Signal:
    """
    :param reference_SWRs:  Only provided for logging, and to be able to reuse
            tune / initial test code.
    """
    outputs, _ = _apply_RNN_chunked(model, LFP, reference_SWRs)
    squished_chunks_np = []
    for unbounded_chunk in outputs:
        squished_chunk: torch.Tensor = torch.sigmoid(unbounded_chunk.squeeze())
        squished_chunks_np.append(squished_chunk.to("cpu").numpy())
    envelope = np.concatenate(squished_chunks_np)
    return Signal(envelope, LFP.fs, units=None)


def _apply_RNN_chunked(
    model: SharpRNN, LFP: Signal, reference_SWRs: SegmentArray
):
    # Cut lfp into (relatively large) chunks, to avoid overloading GPU mem.
    # If run on ALL data of one trimmed recording:
    # 30 minutes * 60 seconds * 1000 samples * 20 channels * 32 bit  = 1.15 GB
    chunks = create_chunks(LFP, reference_SWRs, chunk_duration=120)
    logger = ChunkProgressLogger(num_chunks=len(chunks))
    h = model.get_init_h()
    outputs = []
    for chunk in chunks:
        # noinspection PyUnresolvedReferences
        # (PyCharm can't find "no_grad" in torch.pyi file)
        with torch.no_grad():
            output, h = model.forward(chunk.torch_input, h)
            outputs.append(output)
            chunk_loss = cost_function(output, chunk.torch_target)
            logger.update(chunk, chunk_loss)
    return outputs, logger


@sharp_workflow.task
def select_RNN(
    model_performances: Sequence[Tuple[SharpRNN, float]]
) -> SharpRNN:
    # "zip(*)" unzips: unpack all tuples, and zip together each first element.
    models, performances = zip(*model_performances)
    i = np.array(performances).argmax()
    print(
        f"Model of epoch {i} had best mean sample loss on"
        f" select / internal test set: {performances[i]:.3g}"
    )
    return models[i]


def create_chunks(
    LFP: Signal,
    reference_SWRs: SegmentArray,
    chunk_duration: float,
    randomize_first_chunk_duration: bool = False,
) -> List[Input_TargetPair]:
    """
    The length of the first chunk is randomized. Chunks will therefore start at
    different times for each training epoch. This increases variation when
    iterating multiple times over the same training data set, avoiding
    overfitting to the same data sequences over and over.
    (This is not relevant for testing/validation; but can't hurt either).
    """
    full = Input_TargetPair.create_from(LFP, reference_SWRs)
    if randomize_first_chunk_duration:
        # Make sure first chunk has at least a decent length, to avoid weight
        # updates based on too little information.
        first_chunk_duration = chunk_duration / 2 + uniform(0, chunk_duration)
    else:
        first_chunk_duration = chunk_duration
    chunk_start = 0
    chunk_stop = first_chunk_duration
    chunks = []
    while chunk_start < full.input_sig.duration:
        chunk = Input_TargetPair(
            input_sig=full.input_sig.time_slice(chunk_start, chunk_stop),
            target_sig=full.target_sig.time_slice(chunk_start, chunk_stop),
        )
        chunks.append(chunk)
        chunk_start = chunk_stop
        chunk_stop = chunk_start + chunk_duration
    return chunks


@dataclass
class ChunkProgressLogger:
    num_chunks: int

    def __post_init__(self):
        self.chunk_nr = 0
        self.num_samples_sum = 0
        self.loss_sum = 0
        print("Will print, for each chunk: 1) Mean loss per sample, for chunk")
        print('2) Mean loss per sample, for all chunks up to now ("Running")')

    def update(self, chunk: Input_TargetPair, loss: torch.Tensor):
        self.chunk_nr += 1
        chunk_loss = loss.item()
        chunk_num_samples = chunk.input_sig.num_samples
        mean_sample_loss_chunk = chunk_loss / chunk_num_samples
        self.loss_sum += chunk_loss
        self.num_samples_sum += chunk_num_samples
        info = (
            f"Chunk {self.chunk_nr}",
            f"{self.chunk_nr / self.num_chunks:.1%}",
            f"Loss: {mean_sample_loss_chunk:.3f}",
            f"Running: {self.mean_sample_loss:.3f}",
        )
        # Example message:
        # Chunk 3 | 5.3% | Loss: 0.423 | Running: 0.321
        print(" | ".join(info))

    @property
    def mean_sample_loss(self):
        return self.loss_sum / self.num_samples_sum
