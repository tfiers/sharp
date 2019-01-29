from numpy import linspace

from seaborn import set_hls_values
from sharp.data.files.figure import FigureTarget
from sharp.data.hardcoded.style import fraction, green, paperfig, readable
from sharp.data.types.aliases import subplots
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.plot.misc.F_score import iso_F_line
from sharp.tasks.plot.paper import (
    colors,
    get_sweeps,
    get_tes,
    labels,
    output_dir,
    sweepers,
)
from sharp.tasks.plot.results.PR_and_latency import rank_higher_AUC_lower
from sharp.tasks.plot.util.legend import add_colored_legend

iso_F_color = green


class Plot_PR_Curve(FigureMaker):
    def output(self):
        return FigureTarget(output_dir, "PR-curve")

    def requires(self):
        return sweepers

    def work(self):
        fig, ax = subplots(figsize=paperfig(0.59, 0.497))
        self.setup_ax(ax)
        self.plot_PR(ax)
        self.shade_under_PR_curves(ax)
        self.plot_iso_F_curves(ax)
        self.mark_selected_threshold(ax)
        add_colored_legend(
            ax,
            labels + ("Iso-$F_2$-curves",),
            colors + (iso_F_color,),
            loc=(0.03, 0.03),
        )
        fig.tight_layout()
        self.output().write(fig)

    def mark_selected_threshold(self, ax):
        for te, color in zip(get_tes(), colors):
            ax.plot(te.recall, te.precision, ".", ms=7, c=color, mec="black")

    def plot_iso_F_curves(self, ax):
        R = linspace(0, 1, 1000)
        for F in (0.75, 0.80, 0.85, 0.90, 0.95):
            P = iso_F_line(R, F, beta=2)
            dom = (P > 0) & (P < 1)
            ax.plot(
                R[dom],
                P[dom],
                color=iso_F_color,
                lw=0.8 * readable["lines.linewidth"],
            )
            ax.text(s=f"{F:.0%}", color=iso_F_color, x=1.01, y=P[-1], va="top")

    def plot_PR(self, ax):
        for sweep, color in zip(get_sweeps(), colors):
            ax.plot(sweep.recall, sweep.precision, color=color)

    def shade_under_PR_curves(self, ax):
        tups = zip(get_sweeps(), colors)
        tups = sorted(tups, key=lambda tup: rank_higher_AUC_lower(tup[0]))
        for sweep, color in tups:
            fc = set_hls_values(color, l=0.95)
            ax.fill_between(sweep.recall, sweep.precision, color=fc)

    def setup_ax(self, ax):
        ax.set_xlim(0.17, 1)
        ax.set_ylim(0.17, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.xaxis.set_major_formatter(fraction)
        ax.yaxis.set_major_formatter(fraction)
