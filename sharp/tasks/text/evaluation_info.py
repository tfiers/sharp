from sharp.config.load import final_output_dir
from sharp.data.files.stdlib import DictFile
from sharp.tasks.base import SharpTask
from sharp.tasks.evaluate.sweep import ThresholdSweeper
from sharp.tasks.signal.base import InputDataMixin
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF


class WriteEvalInfo(SharpTask, InputDataMixin):
    envelope_maker = ApplyOnlineBPF()
    sweeper = ThresholdSweeper(envelope_maker=envelope_maker)

    def requires(self):
        return self.input_data_makers + (self.sweeper,)

    def output(self):
        return DictFile(final_output_dir, "evaluation-info")

    def work(self):
        self.output().write(
            {"Lockout time (ms)": self.sweeper.lockout_time * 1000}
        )
