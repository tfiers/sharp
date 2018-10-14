from pandas import DataFrame, concat
from seaborn import JointGrid, kdeplot

from sharp.data.files.figure import FigureTarget
from sharp.tasks.plot.summary.base import SummaryFigureMaker


class PlotLatencyScatter(SummaryFigureMaker):
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
        grid = JointGrid(x=swr_duration, y=delay, data=df, height=5)
        kde_kwargs = dict(legend=False, lw=3)

        for _, data in df.groupby(algo):
            kdeplot(data[swr_duration], ax=grid.ax_marg_x, **kde_kwargs)
            kdeplot(data[delay], ax=grid.ax_marg_y, vertical=True, **kde_kwargs)
            grid.ax_joint.plot(data[swr_duration], data[delay], ".", ms=8)
        fig = grid.fig
        fig.tight_layout()
        self.output().write(fig)
