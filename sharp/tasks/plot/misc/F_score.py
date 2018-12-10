from numpy import linspace, ndarray

from sharp.data.files.figure import FigureTarget
from sharp.data.types.aliases import subplots
from sharp.data.hardcoded.style import paperfig, fraction
from sharp.tasks.plot.base import FigureMaker


class PlotIsoFlines(FigureMaker):
    def output(self):
        return FigureTarget(self.output_dir, "iso-F-lines")

    def work(self):
        fig, axes = subplots(ncols=2, figsize=paperfig(1.1, 0.5))
        eps = 0.001
        lims = (-eps, 1 + eps)
        R = linspace(0, 1, 1000)

        for i, beta in enumerate([1, 2]):
            ax = axes[i]
            color = f"C{i}"
            ax.set_title(f"$F_{beta}$", color=color)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal")
            ax.plot(R, R, "grey", lw=0.5)
            ax.set_xlabel("Recall")
            if i == 0:
                ax.set_ylabel("Precision")
            ax.xaxis.set_major_formatter(fraction)
            ax.yaxis.set_major_formatter(fraction)
            for F in [0.4, 0.6, 0.8, 0.9, 0.95]:
                P = iso_F_line(R, F, beta)
                dom = P > 0
                ax.plot(R[dom], P[dom], color=color)
                ax.text(
                    s=f"{F:.0%}",
                    color=color,
                    x=lims[1] + 0.02,
                    y=P[-1],
                    va="center",
                )

        fig.tight_layout(w_pad=3)
        self.output().write(fig)


def iso_F_line(recall: ndarray, F: float, beta: float) -> ndarray:
    """
    Precision values that yield a constant F-beta score (namely the given
    F-value) in combination with the given recall values.
    """
    return F * recall / ((1 + beta ** 2) * recall - F * beta ** 2)
