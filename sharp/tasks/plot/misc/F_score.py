from numpy import linspace, ndarray

from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.plot.results.base import fraction


class PlotIsoFlines(FigureMaker):
    def output(self):
        return FigureTarget(self.output_dir, "iso-F-lines")

    def work(self):
        fig, axes = subplots(ncols=2, figsize=())
        eps = 0.001
        lims = (-eps, 1 + eps)
        R = linspace(0, 1, 1000)

        for i, beta in enumerate([1, 2]):
            ax = axes[i]
            color = f"C{i}"
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal")
            ax.plot(R, R, "grey", lw=1)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.xaxis.set_major_formatter(fraction)
            ax.yaxis.set_major_formatter(fraction)
            for F in [0.4, 0.6, 0.7, 0.8, 0.9, 0.95]:
                P = iso_F_line(R, F, beta)
                dom = P > 0
                ax.plot(R[dom], P[dom], color=color)
                ax.text(
                    s=f"{F:.0%}", color=color, x=lims[1], y=P[-1], va="center"
                )

        fig.tight_layout()
        self.output().write(fig)


def iso_F_line(recall: ndarray, F: float, beta: float) -> ndarray:
    """
    Precision values that yield a constant F-beta score (namely the given
    F-value) in combination with the given recall values.
    """
    return F * recall / ((1 + beta ** 2) * recall - F * beta ** 2)
