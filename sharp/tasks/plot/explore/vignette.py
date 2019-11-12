from abc import ABC
from typing import Type

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import close
from numpy import arange, ceil

from sharp.data.files.figure import PDF_FigureFile
from sharp.data.types.aliases import subplots
from sharp.tasks.base import WrapperTask
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.plot.util.scalebar import (
    add_time_scalebar,
    add_voltage_scalebar,
)
from sharp.tasks.plot.util.signal import plot_signal
from sharp.tasks.signal.downsample import DownsampleRawRecording
from sharp.tasks.signal.offline_filter import (
    CalcRippleEnvelope,
    CalcSharpWaveEnvelope,
)
from sharp.tasks.signal.raw import SingleRecordingFileTask
from sharp.util.misc import format_duration


output_root = FigureMaker.output_dir / "vignettes"


class PlotVignettes(FigureMaker, SingleRecordingFileTask, ABC):

    WINDOW_DURATION = 3
    # Length of plotted segments, in seconds.
    WINDOW_PERIOD = 60 * 5
    # Seconds between window starts.

    subdir: str = ...
    parent_task_class: Type[SingleRecordingFileTask] = ...

    def requires(self):
        return self.parent_task_class(file_ID=self.file_ID)

    @property
    def output_dir(self):
        return

    def output(self):
        return PDF_FigureFile(output_root / self.subdir, self.file_ID.short_str)

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
                t = format_duration(time_range[0], auto_ms=False)
                ax.set_title(f"{t} since start of recording.")
                fig.tight_layout()
                pdf.savefig(fig)
                close(fig)


class PlotRawVignettes(PlotVignettes):
    subdir = "raw"
    parent_task_class = DownsampleRawRecording


class PlotRippleEnvelopeVignettes(PlotVignettes):
    subdir = "ripple-envelope"
    parent_task_class = CalcRippleEnvelope


class PlotSharpwaveEnvelopeVignettes(PlotVignettes):
    subdir = "sharpwave-envelope"
    parent_task_class = CalcSharpWaveEnvelope


class PlotAllVignettes(WrapperTask, SingleRecordingFileTask):
    def requires(self):
        return (
            PlotRawVignettes(file_ID=self.file_ID),
            PlotRippleEnvelopeVignettes(file_ID=self.file_ID),
            PlotSharpwaveEnvelopeVignettes(file_ID=self.file_ID),
        )
