from sharp.config.load import config
from sharp.data.hardcoded.style import blue, pink
from sharp.tasks.evaluate.sweep import ThresholdSweeper
from sharp.tasks.neuralnet.apply import ApplyRNN
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF
from sharp.tasks.signal.reference import MakeReference

output_dir = FigureMaker.output_dir / "minipaper"

rm = MakeReference(
    mult_detect_SW=config.mult_detect_SW[3],
    mult_detect_ripple=config.mult_detect_ripple[4],
)

sweeper_rnn = ThresholdSweeper(reference_maker=rm, envelope_maker=ApplyRNN())
sweeper_bpf = ThresholdSweeper(
    reference_maker=rm, envelope_maker=ApplyOnlineBPF()
)
color_rnn = pink
color_bpf = blue

sweepers = (sweeper_bpf, sweeper_rnn)
colors = (color_bpf, color_rnn)
labels = ("Band-pass filter", "Recurrent neural net")


def get_sweeps():
    return [sweeper.output().read() for sweeper in sweepers]


def get_tes():
    return [sweep.at_max_F2() for sweep in get_sweeps()]
