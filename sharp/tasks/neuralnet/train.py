"""
Fit a sharp wave-ripple detector to labelled data.
"""

from logging import getLogger
from typing import Iterable, Optional, Sequence

import torch
from luigi import IntParameter

from sharp.data.files.neuralnet import NeuralModelFile
from sharp.tasks.base import SharpTask
from sharp.tasks.neuralnet.base import IOTuple, NeuralNetMixin
from sharp.config.load import config
from sharp.tasks.signal.util import time_to_index
from sharp.util.misc import cached

log = getLogger(__name__)


class TrainRNN(SharpTask, NeuralNetMixin):
    """
    Tunes the weights of a recurrent neural network for one epoch, where an
    epoch is defined as one full pass over the training data set.
    """

    epoch = IntParameter()

    @property
    def is_initial_epoch(self) -> bool:
        return self.epoch == 0

    @property
    def prev_epoch_trainer(self) -> Optional["TrainRNN"]:
        if self.epoch == 0:
            return None
        else:
            return TrainRNN(epoch=self.epoch - 1)

    def requires(self):
        dependencies = self.input_data_makers
        if not self.is_initial_epoch:
            dependencies += (self.prev_epoch_trainer,)
        return dependencies

    def output(self) -> NeuralModelFile:
        return NeuralModelFile(
            self.output_dir, f"model-{self.epoch}", self.get_model_prototype()
        )

    @property
    def output_dir(self):
        return super().output_dir / "epochs"

    def work(self):
        if not self.is_initial_epoch:
            self.model = self.prev_epoch_trainer.output().read()
        self.tune_weights()
        self.output().write(self.model)

    def tune_weights(self):
        chunks = self.to_chunks([self.io_tuple_train])
        total_training_loss = 0
        # Get a random initial hidden state.
        h0 = self.model.get_init_h()
        for i, (input_chunk, target_chunk) in enumerate(chunks):
            # Run chunk of input signal through RNN.
            output, hn = self.model.forward(input_chunk, h0)
            # Evaluate cost function (calculate 'loss') by comparing output with
            # the reference signal.
            loss = self.cost_function(output, target_chunk)
            # Bookkeeping
            progress = i / self.num_training_chunks
            log.info(f"Epoch {self.epoch} | {progress:.1%}")
            total_training_loss += loss.item()
            # Empty the partial derivative accumulator of each weight.
            self.model.zero_grad()
            # Calculate partial derivative of loss wrt. each model weight.
            loss.backward()
            # Perform gradient descent step.
            self.weight_updater.step()
            # Iterate.
            h0 = hn.detach()
            # "Detaches the Tensor from the graph that created it, making it a
            # leaf."  We do not want to calculate partial derivatives of weights
            # before this time step. (Or do we?? Todo: test / work out).

        mean_chunk_loss = total_training_loss / self.num_training_chunks
        log.info(
            f"Finished training epoch {self.epoch} "
            f"| Mean loss per training chunk: {mean_chunk_loss:.4g}"
        )

    @property
    @cached
    def weight_updater(self) -> torch.optim.Optimizer:
        """
        When `weight_updater.step()` is called, the weights of `self.model` are
        updated, based on the 'gradient' values stored along each weight. Each
        such value is the partial derivative of the loss function with respect
        to that weight. (We thus perform gradient descent).
        """
        # The gradient descent algorithm 'AdaMax' keeps track of exponentially
        # weighted and normalised moving averages of the partial loss function
        # derivatives, and uses these to update the parameters.
        return torch.optim.Adamax(self.model.parameters())

    def to_chunks(self, io_tuples: Sequence[IOTuple]) -> Iterable[IOTuple]:
        """
        Cuts up each IO tuple into short chunks, and lumps all chunks together.
        """
        for input_slice, target_slice in io_tuples:
            # Split along time dimension.
            num_chunks = input_slice.shape[1] // self.chunk_size
            chunk_tups = zip(
                torch.chunk(input_slice, num_chunks, dim=1),
                torch.chunk(target_slice, num_chunks, dim=1),
            )
            for tup in chunk_tups:
                yield tup

    @property
    def chunk_size(self) -> int:
        """ Number of samples per chunk. """
        size = time_to_index(
            config.chunk_duration, self.reference_channel_full.fs
        )
        return int(size)

    @property
    @cached
    def num_training_chunks(self) -> int:
        """ Total number of training chunks, over all traininig IO tuples."""
        num_training_samples = self.input_signal_train_proper.num_samples
        return int(num_training_samples // self.chunk_size)
