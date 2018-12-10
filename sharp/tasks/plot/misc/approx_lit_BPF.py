from matplotlib.axes import Axes
from numpy import linspace

from sharp.config.load import final_output_dir
from sharp.data.files.figure import FigureTarget
from sharp.data.hardcoded.filters.base import LTIRippleFilter
from sharp.data.hardcoded.filters.literature import (
    DuttaOriginal,
    DuttaReplica,
    EgoStengelOriginal,
    EgoStengelReplica,
    FalconOriginal,
    FalconReplica,
)
from sharp.data.hardcoded.filters.util import dB, gain, group_delay
from sharp.data.hardcoded.style import paperfig
from sharp.data.types.aliases import subplots
from sharp.tasks.base import SharpTask
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.plot.util.legend import add_colored_legend


class PlotAllOnlineBPFReplications(SharpTask):
    def requires(self):
        return (EgoStengel(), Dutta(), Falcon())


class PlotOnlineBPFReplication(FigureMaker):

    filter_original: LTIRippleFilter = ...
    filter_replica: LTIRippleFilter = ...

    def output(self):
        dirr = final_output_dir / "online-BPF-replicas"
        filename = self.__class__.__name__
        return FigureTarget(dirr, filename)

    def work(self):
        fig, axes = subplots(
            nrows=2, ncols=2, figsize=paperfig(width=1.2, height=0.55)
        )
        ax_top_left: Axes = axes[0, 0]
        ax_top_right: Axes = axes[0, 1]
        ax_bottom_left: Axes = axes[1, 0]
        ax_bottom_right: Axes = axes[1, 1]
        ax_top_right.remove()
        ax_bottom_left.set_xlabel("Frequency (Hz)")
        ax_bottom_right.set_xlabel("Frequency (Hz)")
        ax_gain_dB = ax_top_left
        ax_gain = ax_bottom_left
        ax_grpdelay = ax_bottom_right
        ax_gain.set_ylabel("Gain")
        ax_gain_dB.set_ylabel("Gain (dB)")
        ax_gain_dB.set_ylim(-63, 4)
        ax_grpdelay.set_ylabel("Group delay (ms)")
        # Force zero-line in view:
        ax_grpdelay.axhline(y=0, color="none")
        f_max = 500
        margin = 10  # To avoid phase discontinutities. In Hz.
        f = linspace(margin, f_max - margin, 10000)
        for filta in (self.filter_original, self.filter_replica):
            H = filta.freqresp(f)
            g = gain(H)
            ax_gain.plot(f, g)
            ax_gain_dB.plot(f, dB(g))
            ax_grpdelay.plot(f, group_delay(H, f))
        add_colored_legend(
            fig,
            (self.label_original, self.label_replica),
            loc="lower left",
            bbox_to_anchor=(0.5, 0.6),
        )
        fig.tight_layout(w_pad=3)
        self.output().write(fig)

    @property
    def label_original(self):
        return f"Original ($f_s = {self.filter_original.fs:.0f}$ Hz)"

    @property
    def label_replica(self):
        return f"Replication ($f_s = {self.filter_replica.fs:.0f}$ Hz)"


class EgoStengel(PlotOnlineBPFReplication):
    filter_original = EgoStengelOriginal()
    filter_replica = EgoStengelReplica()
    label_original = "Original (continuous time)"


class Dutta(PlotOnlineBPFReplication):
    filter_original = DuttaOriginal()
    filter_replica = DuttaReplica()


class Falcon(PlotOnlineBPFReplication):
    filter_original = FalconOriginal()
    filter_replica = FalconReplica()
