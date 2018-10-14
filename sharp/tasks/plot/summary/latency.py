from matplotlib.pyplot import figure
from matplotlib.ticker import PercentFormatter
from pandas import DataFrame, concat
from seaborn import set_hls_values

from raincloud import distplot
from sharp.data.files.figure import FigureTarget
from sharp.tasks.plot.summary.base import SummaryFigureMaker


class PlotLatency(SummaryFigureMaker):
    """
    Plots the distribution of relative detection delays, at a fixed recall
    value.
    """

    def output(self):
        return FigureTarget(self.output_dir, "Relative-delays")

    def run(self):
        fig = figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)
        self.make_distplots(ax)
        self.set_xaxis(ax)
        self.remove_text(ax)
        fig.tight_layout()
        self.output().write(fig)

    def set_xaxis(self, ax):
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    def remove_text(self, ax):
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticklabels([])

    def make_distplots(self, ax):
        data_header = "Detection delay"
        algo_header = "Algorithm"
        df_dicts = (
            {data_header: sweep.best.rel_delays, algo_header: title}
            for sweep, title in zip(self.threshold_sweeps, self.titles)
        )
        df: DataFrame = concat(DataFrame(dic) for dic in df_dicts)
        if not df.empty:
            palette = [set_hls_values(c, l=0.6) for c in self.colors]
            distplot(
                data=df,
                x=data_header,
                y=algo_header,
                ax=ax,
                bw=0.2,
                palette=palette,
                ms=6,
                move=0.2,
                offset=0.1,
            )
