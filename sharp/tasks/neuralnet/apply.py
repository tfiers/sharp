import torch
from numpy import ndarray

from sharp.data.types.aliases import TorchArray
from sharp.data.types.neuralnet import RNN
from sharp.data.types.signal import Signal
from sharp.tasks.neuralnet.base import NeuralNetMixin
from sharp.tasks.neuralnet.select import SelectBestRNN
from sharp.tasks.signal.base import EnvelopeMaker


class ApplyRNN(EnvelopeMaker, NeuralNetMixin):

    title = "Recurrent neural network"
    output_filename = "neural-net"

    model_selector = SelectBestRNN()

    def requires(self):
        return super().requires() + (self.model_selector,)

    def work(self):
        with torch.no_grad():
            inputt = self.as_model_io(self.reference_channel_full.as_matrix())
            model: RNN = self.model_selector.output().read()
            h0 = model.get_init_h()
            output, _ = model(inputt, h0)
            # Cannot use torch.nn.functional.sigmoid (deprecated).
            envelope: TorchArray = torch.sigmoid(output.squeeze())
            envelope_cpu = envelope.to("cpu")
            envelope_numpy: ndarray = envelope_cpu.numpy()
            sig = Signal(envelope_numpy, self.reference_channel_full.fs)
            self.output().write(sig)
