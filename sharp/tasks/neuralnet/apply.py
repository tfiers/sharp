from logging import getLogger

import torch
from numpy import array_split, concatenate

from sharp.data.types.aliases import TorchArray
from sharp.data.types.neuralnet import RNN
from sharp.data.types.signal import Signal
from sharp.tasks.neuralnet.base import NeuralNetMixin
from sharp.tasks.neuralnet.select import SelectBestRNN
from sharp.tasks.signal.base import EnvelopeMaker

log = getLogger(__name__)


class ApplyRNN(EnvelopeMaker, NeuralNetMixin):

    title = "Recurrent neural network"
    output_filename = "neural-net"
    seconds_per_chunk = 10

    model_selector = SelectBestRNN()

    def requires(self):
        return super().requires() + (self.model_selector,)

    def work(self):
        envelope_chunks = []
        inputt = self.multichannel_full.as_matrix()
        # Full input signal is too big for GPU memory.
        # Thus: split input, pass through h, concatenate results
        num_chunks = inputt.duration // self.seconds_per_chunk
        input_chunks = array_split(inputt, num_chunks, axis=inputt.time_axis)
        model: RNN = self.model_selector.output().read()
        h = model.get_init_h()
        for i, input_chunk in enumerate(input_chunks):
            log.info(f"Transforming chunk {i} of {num_chunks}")
            with torch.no_grad():
                input_torch = self.as_model_io(input_chunk)
                output, h = model(input_torch, h)
                # Cannot use torch.nn.functional.sigmoid (deprecated).
                envelope: TorchArray = torch.sigmoid(output.squeeze())
                envelope_cpu = envelope.to("cpu")
                envelope_numpy = envelope_cpu.numpy()
            envelope_chunks.append(envelope_numpy)
        envelope_full = concatenate(envelope_chunks)
        sig = Signal(envelope_full, inputt.fs)
        self.output().write(sig)
