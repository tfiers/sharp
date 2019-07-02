import torch

from farao import File
from sharp.data.types.neuralnet import RNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Memory location of model (either at a GPU, or at the CPU).


class RNNFile(File):

    extension = ".pytorch"

    def read(self, architecture: RNN) -> RNN:
        state_dict = torch.load(self.path_string, map_location=device)
        architecture.load_state_dict(state_dict)
        return architecture

    def write(self, model: RNN):
        torch.save(model.state_dict(), self.path_string)
