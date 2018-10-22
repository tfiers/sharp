from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from matplotlib.ticker import FuncFormatter
from numpy import arange, exp, linspace, log10, ndarray
from sklearn.neighbors import KernelDensity

from raincloud import distplot
from sharp.data.files.figure import FigureTarget
from sharp.tasks.base import SharpTask
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.plot.style import seaborn_colours
from sharp.tasks.plot.util.scalebar import add_scalebar
from sharp.tasks.signal.base import InputDataMixin


class PlotRecordingSummaries(SharpTask):
    def requires(self):
        return (PlotSWRDurations(), PlotInterSWRIntervals(), PlotSWRLocations())


class RecordingSummary(FigureMaker, InputDataMixin):
    output_dir = FigureMaker.output_dir / "recording-summary"

    def requires(self):
        self.input_data_makers


class PlotSWRDurations(RecordingSummary):
    def output(self):
        return FigureTarget(self.output_dir, "SWR-durations")

    def run(self):
        fig, ax = subplots(figsize=(8, 4))  # type: Figure, Axes
        durations = self.reference_segs_all.duration
        distplot(1e3 * durations, ax=ax, palette=seaborn_colours)
        ax.set_xlabel("SWR duration (ms)")
        fig.tight_layout()
        self.output().write(fig)


class PlotInterSWRIntervals(RecordingSummary):
    def output(self):
        return FigureTarget(self.output_dir, "inter-SWR-intervals")

    def run(self):
        intervals = self.reference_segs_all.intervals
        fig, ax = subplots(figsize=(8, 4))  # type: Figure, Axes
        distplot(log10(intervals), ax=ax)
        ax.set_xticks(arange(-2, 3, dtype=float))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{10**x:.3g}")
        )
        ax.set_xlabel("Inter-SWR interval (s)")
        fig.tight_layout()
        self.output().write(fig)


class PlotSWRLocations(RecordingSummary):
    def output(self):
        return FigureTarget(self.output_dir, "SWR-locations")

    def run(self):
        pos = self.reference_segs_all.center
        sig_length = self.reference_channel_full.duration
        kde = KernelDensity(bandwidth=0.2)
        kde.fit(as_data_matrix(pos))
        fig, ax = subplots(figsize=(14, 2.5))  # type: Figure, Axes
        t = linspace(0, sig_length, num=4000)
        log_density = kde.score_samples(as_data_matrix(t))
        density = exp(log_density)
        t_min = t / 60
        ax.plot(t_min, density)
        ax.fill_between(t_min, density, alpha=0.3)
        ax.set_xlabel("Time (min)")
        f = 2
        add_scalebar(ax, "v", f, f"{f} Hz", pos_along=0, pos_across=0.03)
        ax.set_yticks([])
        fig.tight_layout()
        self.output().write(fig)


def as_data_matrix(vec: ndarray) -> ndarray:
    return vec.reshape((-1, 1))
