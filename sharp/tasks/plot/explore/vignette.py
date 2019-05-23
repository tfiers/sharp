from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import close
from numpy import arange, ceil

from sharp.data.files.figure import PDF_FigureFile
from sharp.data.types.aliases import subplots
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.plot.util.scalebar import (
    add_time_scalebar,
    add_voltage_scalebar,
)
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

    output_dir = FigureMaker.output_dir / "raw-vignettes"

    def output(self):
        return PDF_FigureFile(self.output_dir, self.file_ID.short_str)

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
        with PdfPages(self.output().path_string) as pdf:
            for time_range in self.time_ranges:
                fig, ax = subplots()
                plot_signal(self.signal, time_range, ax=ax, time_grid=False)
                add_time_scalebar(ax, 500, "ms", pos_across=-0.04)
                add_voltage_scalebar(ax, pos_across=-0.04)
                fig.tight_layout()
                pdf.attach_note(f"Time range: {time_range}")
                pdf.savefig(fig)
                close(fig)
