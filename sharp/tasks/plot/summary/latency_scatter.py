from matplotlib.figure import Figure
from pandas import DataFrame, concat
from seaborn import JointGrid, kdeplot

from sharp.data.files.figure import FigureTarget
from sharp.tasks.plot.summary.base import MultiEnvelopeSummary
from sharp.tasks.plot.util.legend import add_colored_legend


class PlotLatencyScatter(MultiEnvelopeSummary):
    def output(self):
        return FigureTarget(self.output_dir, "latency-scatter")

    def run(self):
        swr_duration = "SWR duration (ms)"
        delay = "Detection latency (ms)"
        algo = "Algorithm"
        df_dicts = (
            {
                swr_duration: 1000
                * sweep.best.detected_reference_segs.duration,
                delay: 1000 * sweep.best.abs_delays,
                algo: title,
            }
            for sweep, title in zip(self.threshold_sweeps, self.titles)
        )
        df: DataFrame = concat(DataFrame(dic) for dic in df_dicts)
        grid = JointGrid(x=swr_duration, y=delay, data=df, height=8)
        kde_kwargs = dict(legend=False, lw=3)
        scatter_kwargs = dict(
            linestyle="none", marker=".", markeredgecolor="none", markersize=6
        )
        for title in self.titles:
            data = df[getattr(df, algo) == title]
            kdeplot(data[swr_duration], ax=grid.ax_marg_x, **kde_kwargs)
            kdeplot(data[delay], ax=grid.ax_marg_y, vertical=True, **kde_kwargs)
            grid.ax_joint.plot(
                data[swr_duration], data[delay], **scatter_kwargs
            )
        fig = grid.fig  # type: Figure
        add_colored_legend(fig, self.titles, self.colors)
        fig.tight_layout()
        self.output().write(fig)
