from sharp.config.load import final_output_dir
from sharp.data.files.stdlib import DictFile
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.online_bpf import ApplyOnlineBPF


class WriteOnlineBPFInfo(SharpTask):

    filtertask = ApplyOnlineBPF()

    def requires(self):
        return self.filtertask

    def output(self):
        return DictFile(final_output_dir, "online-BPF")

    def work(self):
        filta = self.filtertask.ripple_filter
        b, a = filta.tf
        self.output().write(
            {
                "fs": filta.fs,
                "numerator-b": b.tolist(),
                "denominator-a": a.tolist(),
                "numtaps": len(a),
            }
        )
