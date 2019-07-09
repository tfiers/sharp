from dataclasses import dataclass
from random import uniform
from typing import Callable, Iterable, Mapping, Sequence

import torch
import torch.nn
import torch.optim

from sharp.datatypes.neuralnet import Input_TargetPair, SharpRNN
from sharp.datatypes.segments import SegmentArray
from sharp.datatypes.signal import Signal
from sharp.main import config, sharp_workflow


@sharp_workflow.task
def get_init_model(LFP: Signal) -> SharpRNN:
    """ Initialize a fresh model for the first epoch. """
    hyperparams = config.RNN_hyperparams
    # todo: next line doesn't work: that's a Task
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
    chunks = list(get_training_chunks(LFP, reference_SWRs))
    # The gradient descent algorithm 'AdaMax' keeps track of exponentially
    # weighted and normalised moving averages of the partial derivatives
    # (pdv's) of the loss function wrt. each weight, and uses these to update
    # the weights.
    weight_updater = torch.optim.Adamax(model.parameters())
    h = model.get_init_h()
    time_since_last_detach = 0
    logger = TrainProgressLogger(num_chunks=len(chunks))
    print("Will print, for each chunk: 1) Mean loss per sample, for chunk")
    print('2) Mean loss per sample, for all chunks up to now ("Running")')
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
    # todo:
    raise UserWarning("Did not use train-part only of LFP and refsegs")


def get_training_chunks(
    LFP: Signal, reference_SWRs: SegmentArray
) -> Iterable[Input_TargetPair]:
    """
    The length of the first chunk is randomized. Chunks will therefore start at
    different times for each training epoch. This increases variation when
    iterating multiple times over the same training data set, avoiding
    overfitting to the same data sequences over and over.
    """
    full = Input_TargetPair.create_from(LFP, reference_SWRs)
    chunk_duration = config.RNN_training.chunk_duration
    # Make sure first chunk has at least a decent length, to avoid weight
    # updates based on too little information.
    first_chunk_duration = chunk_duration / 2 + uniform(0, chunk_duration)
    chunk_start = 0
    chunk_stop = first_chunk_duration
    while chunk_start < full.input_sig.duration:
        yield Input_TargetPair(
            input_sig=full.input_sig.time_slice(chunk_start, chunk_stop),
            target_sig=full.target_sig.time_slice(chunk_start, chunk_stop),
        )
        chunk_start = chunk_stop
        chunk_stop = chunk_start + chunk_duration


@dataclass
class TrainProgressLogger:
    num_chunks: int

    def __post_init__(self):
        self.chunk_nr = 0
        self.running_sum_num_samples = 0
        self.running_sum_loss = 0

    def update(self, chunk: Input_TargetPair, loss: torch.Tensor):
        self.chunk_nr += 1
        chunk_loss = loss.item()
        chunk_num_samples = chunk.input_sig.num_samples
        mean_sample_loss_chunk = chunk_loss / chunk_num_samples
        self.running_sum_loss += chunk_loss
        self.running_sum_num_samples += chunk_num_samples
        mean_sample_loss_running = (
            self.running_sum_loss / self.running_sum_num_samples
        )
        info = (
            f"Chunk {self.chunk_nr}",
            f"{self.chunk_nr / self.num_chunks:.1%}",
            f"Loss: {mean_sample_loss_chunk:.3f}",
            f"Running: {mean_sample_loss_running:.3f}",
        )
        # Example message:
        # Chunk 3 | 5.3% | Loss: 0.423 | Running: 0.321
        print(" | ".join(info))


def test_RNN_performance(
    model: SharpRNN, LFP: Signal, reference_SWRs: SegmentArray
):

    loss = cost_function()


def select_RNN(
    models: Sequence[SharpRNN], performances: Sequence[float]
) -> SharpRNN:
    ...
