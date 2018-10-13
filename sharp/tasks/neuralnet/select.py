from logging import getLogger
from typing import Iterable

import torch
from luigi import IntParameter

from sharp.data.types.neuralnet import RNN
from sharp.data.files.neuralnet import NeuralModelFile
from sharp.data.files.numpy import NumpyArrayFile
from sharp.data.files.stdlib import FloatFile
from sharp.tasks.neuralnet.base import NeuralNetTask
from sharp.tasks.neuralnet.config import neural_net_config
from sharp.tasks.neuralnet.train import TrainRNN

log = getLogger(__name__)


class CalcValidLoss(NeuralNetTask):
    """
    Run network on (held out) validation data, and evaluate cost function.
    """

    epoch = IntParameter()

    @property
    def trainer(self):
        return TrainRNN(epoch=self.epoch)

    def requires(self):
        return super().requires() + (self.trainer,)

    def output(self):
        model_file = self.trainer.output()
        return FloatFile(
            directory=model_file.parent,
            filename=f"{model_file.stem}.valid-loss",
        )

    def run(self):
        with torch.no_grad():
            valid_tuples = self.make_io_tuple(self.valid_segs)
            model: RNN = self.trainer.output().read()
            loss = 0
            for input_slice, target_slice in valid_tuples:
                h0 = model.get_init_h()
                output, _ = model.forward(input_slice, h0)
                loss += self.cost_function(output, target_slice).item()

            self.output().write(loss)
            log.info(f"Validation loss at epoch {self.epoch}: {loss:.4g}")


class GatherValidLosses(NeuralNetTask):
    def requires(self) -> Iterable[CalcValidLoss]:
        for epoch in range(neural_net_config.num_epochs):
            yield CalcValidLoss(epoch=epoch)

    def output(self):
        return NumpyArrayFile(self.output_dir, "validation-losses")

    def run(self):
        validation_losses = [
            calc_valid_task.output().read()
            for calc_valid_task in self.requires()
        ]
        self.output().write(validation_losses)


class SelectBestRNN(NeuralNetTask):
    """
    Selects the trained RNN with the best generalization performance.
    Generalization performance is estimated by running each trained network
    on a held-out validation set.
    """

    valid_loss_gatherer = GatherValidLosses()

    def requires(self):
        return self.valid_loss_gatherer

    def output(self) -> NeuralModelFile:
        return NeuralModelFile(
            self.output_dir, "best-model", self.get_model_prototype()
        )

    def run(self):
        # Get index of model with lowest validation loss.
        valid_losses = self.input().read()
        best_epoch = int(valid_losses.argmin())
        log.info(f"Best estim. general. perf. at epoch {best_epoch}.")
        # Use this index to copy model file.
        valid_loss_calculators = list(self.valid_loss_gatherer.requires())
        best_valid_loss_calculator = valid_loss_calculators[best_epoch]
        best_model = best_valid_loss_calculator.trainer.output().read()
        self.output().write(best_model)
