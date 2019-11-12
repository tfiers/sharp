from abc import ABC

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import close
from numpy import concatenate, linspace
from sklearn.neighbors import KernelDensity

from sharp.config.load import config
from sharp.data.files.figure import FigureTarget
from sharp.data.hardcoded.style import paperfig
from sharp.data.types.aliases import subplots
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.signal.mountains import (
    DetectRipples,
    DetectSharpwaves,
    DetectMountains,
)


class PlotMountainDensity(FigureMaker, ABC):
    def Hahaha(self):
        return [
            (DetectRipples(file_ID=fid), DetectSharpwaves(file_ID=fid))
            for fid in config.raw_data
        ]

    def output(self):
        return FigureTarget(
            directory=self.output_dir, filename="mountain-density"
        )

    def work(self):
        MINUTES_PER_PAGEWIDTH = 30
        SAMPLE_PERIOD = 10  # seconds
        with PdfPages(self.output().path_string) as pdf:
            for ripple_detector, sharpwave_detector in self.Hahaha():
                duration = ripple_detector.envelope.duration
                pagewidths = duration / (60 * MINUTES_PER_PAGEWIDTH)
                fig, (top, btm) = subplots(
                    nrows=2, figsize=paperfig(pagewidths)
                )
                num_samples = duration // SAMPLE_PERIOD
                time_grid = linspace(0, duration, num_samples)
                plot_density(ripple_detector, top, time_grid)
                plot_density(sharpwave_detector, btm, time_grid)
                btm.set_xlabel("Recording time (minutes)")
                btm.set_ylabel("Mountains per second")
                pdf.savefig(fig)
                close(fig)


def plot_density(detector: DetectMountains, ax, time_grid):
    segs_list = detector.output().read()
    # Yeet mountains together over channels
    seg_starts = concatenate([segs.start for segs in segs_list])
    kde = KernelDensity(bandwidth=10)
    kde.fit(as_data_matrix(seg_starts))
    density = kde.score_samples(as_data_matrix(time_grid))
    ax.plot(time_grid / 60, density)


def as_data_matrix(vector):
    return vector[:, None]
