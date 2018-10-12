from pathlib import Path
from typing import Union

import torch

from sharp.data.types.aliases import NeuralModel
from sharp.data.files.base import FileTarget


class NeuralModelFile(FileTarget):

    extension = ".weights"
    _model: NeuralModel = ...

    def __new__(
        cls, directory: Union[Path, str], filename: str, model: NeuralModel
    ):
        instance: NeuralModelFile = super().__new__(cls, directory, filename)
        instance._model = model
        return instance

    def read(self) -> NeuralModel:
        state_dict = torch.load(self.path_string)
        self._model.load_state_dict(state_dict)
        return self._model

    def write(self, model: NeuralModel):
        self._model = model
        torch.save(self._model.state_dict(), self.path_string)
