from warnings import warn

from numpy import abs, max

from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.tasks.multilin.apply import SpatiotemporalConvolution
from sharp.tasks.plot.results.base import MultiEnvelopeFigureMaker


class PlotWeights(MultiEnvelopeFigureMaker):
    @property
    def convolvers(self):
        return [
            em
            for em in self.envelope_makers
            if isinstance(em, SpatiotemporalConvolution)
        ]

    @property
    def colors(self):
        return [
            color
            for color, em in zip(super().colors, self.envelope_makers)
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

    def work(self):
        tups = zip(self.trainers, self.convolvers, self.colors, self.output())
        for trainer, convolver, color, filetarget in tups:
            fig, ax = subplots(figsize=(5 + convolver.num_delays / 3, 5))
            GEVec = trainer.output().read()
            signal = trainer.multichannel_train
            num_channels = signal.num_channels
            weights = shape_GEVec(GEVec, num_channels)
            wmax = max(abs(weights))
            image = ax.imshow(
                weights, origin="lower", cmap="PiYG", vmin=-wmax, vmax=wmax
            )
            cbar = fig.colorbar(image, ax=ax, shrink=0.6)
            cbar.set_label("Weight")
            ax.set_xticks(convolver.delays)
            ax.set_yticks(convolver.channels)
            ax.set_xlim(ax.get_xlim()[::-1])
            if signal.fs != 1000:
                warn('Delay axis scale "(ms)" in GEVec plot is not correct.')
            ax.set_xlabel("Delay (ms)")
            ax.set_ylabel("Channel")
            title = f"GEVec\n({convolver.filename})"
            ax.set_title(title, color=color)
            ax.grid(False)
            fig.tight_layout()
            filetarget.write(fig)


def shape_GEVec(GEVec, num_channels):
    num_delays = GEVec.size // num_channels
    shape = (num_delays, num_channels)
    weights_shaped = GEVec.reshape(shape).transpose()
    return weights_shaped
