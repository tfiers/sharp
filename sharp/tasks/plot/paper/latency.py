from matplotlib.axes import Axes
from pandas import concat, DataFrame

from raincloud import distplot
from sharp.data.files.figure import FigureTarget
from sharp.data.hardcoded.style import paperfig, fraction
from sharp.data.types.aliases import subplots
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.plot.paper import sweepers, output_dir, get_tes, colors


class PlotLatency(FigureMaker):
    def requires(self):
        return sweepers

    def output(self):
        return FigureTarget(output_dir, "latency")

    def work(self):
        fig, (ax_top, ax_btm) = subplots(nrows=2, figsize=paperfig(0.60, 0.57))
        ax_btm: Axes
        te_bpf, te_rnn = get_tes()
        delays = concat(
            [
                DataFrame(
                    dict(
                        Absolute=te_rnn.abs_delays,
                        Relative=te_rnn.rel_delays,
                        Algorithm="RNN",
                    )
                ),
                DataFrame(
                    dict(
                        Absolute=te_bpf.abs_delays,
                        Relative=te_bpf.rel_delays,
                        Algorithm="BPF",
                    )
                ),
            ]
        )
        delays.Absolute *= 1000
        delays = delays[delays.Absolute < 100]
        kwargs = dict(
            data=delays,
            y="Algorithm",
            alpha_dot=0.4,
            palette=colors,
            order=("BPF", "RNN"),
            width_kde=0.8,
            width_box=0.1,
            jitter=0.12,
            ms=2,
        )
        distplot(x="Absolute", ax=ax_top, **kwargs)
        distplot(x="Relative", ax=ax_btm, **kwargs)
        ax_btm.xaxis.set_major_formatter(fraction)
        for ax in (ax_top, ax_btm):
            ax.set_yticklabels([])
            ax.set_ylabel("")
            ax.set_xlabel("")
        # ax_top.set_xlabel("Absolute detection latency (ms)")
        # ax_btm.set_xlabel("Relative detection latency")
        fig.tight_layout(h_pad=4)
        self.output().write(fig)
