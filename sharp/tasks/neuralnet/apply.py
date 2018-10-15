import torch
from numpy.core.multiarray import ndarray

from sharp.data.files.numpy import SignalFile
from sharp.data.types.aliases import TorchArray
from sharp.data.types.neuralnet import RNN
from sharp.data.types.signal import Signal
from sharp.tasks.neuralnet.base import NeuralNetTask
from sharp.tasks.neuralnet.select import SelectBestRNN
from sharp.tasks.signal.base import EnvelopeMaker


class ApplyRNN(NeuralNetTask):

    model_selector = SelectBestRNN()

    def requires(self):
        return super().requires() + (self.model_selector,)

    def output(self):
        return SignalFile(EnvelopeMaker.output_dir, "neural-net")

    def run(self):
        with torch.no_grad():
            inputt = self.as_model_io(self.input_signal_all.as_matrix())
            model: RNN = self.model_selector.output().read()
            h0 = model.get_init_h()
            output, _ = model(inputt, h0)
            # Cannot use torch.nn.functional.sigmoid (deprecated).
            envelope: TorchArray = torch.sigmoid(output.squeeze())
            envelope_cpu = envelope.to("cpu")
            envelope_numpy: ndarray = envelope_cpu.numpy()
            sig = Signal(envelope_numpy, self.input_signal_all.fs)
            self.output().write(sig)
