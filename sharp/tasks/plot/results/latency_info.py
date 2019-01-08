from sharp.data.files.stdlib import DictFile
from sharp.tasks.plot.results.base import MultiEnvelopeFigureMaker


class WriteLatencyInfo(MultiEnvelopeFigureMaker):
    def output(self):
        return DictFile(self.output_dir, "latency-info")

    def work(self):
        self.output().write(
            {
                f"Median absolute delay (ms) for {title}": sweep.at_max_F1().abs_delays_median
                * 1000
                for title, sweep in zip(self.titles, self.threshold_sweeps)
            }
        )
