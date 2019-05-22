import matplotlib.pyplot as plt
from numpy import arange, ceil
from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.plot.util.signal import plot_signal
from sharp.tasks.signal.downsample import DownsampleRawRecording
from sharp.tasks.signal.raw import SingleRecordingFileTask


class PlotVignettes(FigureMaker, SingleRecordingFileTask):

    WINDOW_DURATION = 3
    # Length of plotted segments, in seconds.
    WINDOW_PERIOD = 60 * 5
    # Seconds between window starts.

    def requires(self):
        return DownsampleRawRecording(file_ID=self.file_ID)

    @property
    def output_dir(self):
        return FigureMaker.output_dir / "raw-vignettes" / self.file_ID.short_str

    def output(self):
        for start, stop in self.time_ranges:
            filename = f"{start:.1f}--{stop:.1f}"
            yield FigureTarget(self.output_dir, filename)

    @property
    def signal(self):
        return self.requires().output().read()

    @property
    def time_ranges(self):
        duration = self.signal.duration
        num_windows = ceil(duration / self.WINDOW_PERIOD)
        starts = self.WINDOW_PERIOD * arange(num_windows)
        return [(t0, t0 + self.WINDOW_DURATION) for t0 in starts]

    def work(self):
        for time_range, output in zip(self.time_ranges, self.output()):
            fig, ax = subplots()
            plot_signal(self.signal, time_range, ax=ax)
            output.write(fig)
            plt.close()
