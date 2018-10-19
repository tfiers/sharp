from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from numpy import abs, max

from sharp.data.files.figure import FigureTarget
from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.plot.summary.base import MultiEnvelopeSummary


class PlotWeights(MultiEnvelopeSummary):
    @property
    def convolvers(self):
        return [
            em
            for em in self.envelope_makers
            if isinstance(em, SpatiotemporalConvolution)
        ]

    @property
    def trainers(self):
        return [c.trainer for c in self.convolvers]

    def output(self):
        return [
            FigureTarget(self.output_dir / "GEVecs", t.filename)
            for t in self.trainers
        ]

    def run(self):
        for trainer, filetarget in zip(self.trainers, self.output()):
            fig, ax = subplots(figsize=(4, 5))  # type: Figure, Axes
            GEVec = trainer.output().read()
            num_channels = trainer.multichannel_train.num_channels
            weights = shape_GEVec(GEVec, num_channels)
            wmax = max(abs(weights))
            cax = ax.imshow(
                weights, origin="lower", cmap="PiYG", vmin=-wmax, vmax=wmax
            )
            cbar = fig.colorbar(cax)
            cbar.set_label("Weight")
            ax.set_xticks(trainer.delays)
            ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_xlabel("Delay")
            ax.set_ylabel("Channel")
            ax.grid(False)
            fig.tight_layout()
            filetarget.write(fig)


def shape_GEVec(GEVec, num_channels):
    num_delays = GEVec.size // num_channels
    shape = (num_delays, num_channels)
    weights_shaped = GEVec.reshape(shape).transpose()
    return weights_shaped
