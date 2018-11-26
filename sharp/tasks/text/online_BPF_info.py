from sharp.config.load import final_output_dir
from sharp.data.files.stdlib import DictFile
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF


class SaveBPFinfo(SharpTask):

    filtertask = ApplyOnlineBPF()

    def requires(self):
        return self.filtertask

    def output(self):
        return DictFile(final_output_dir, "online-BPF")

    def work(self):
        b, a = self.filtertask.coeffs
        self.output().write(
            {
                "fs": self.filtertask.input_signal.fs,
                "numerator-b": b.tolist(),
                "denominator-a": a.tolist(),
                "order": len(a),
            }
        )
