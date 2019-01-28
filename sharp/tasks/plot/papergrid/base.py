from numpy import max

from sharp.config.load import config
from sharp.data.files.figure import FigureTarget
from sharp.data.hardcoded.style import paperfig
from sharp.data.types.aliases import subplots
from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.tasks.base import TaskParameter
from sharp.tasks.evaluate.sweep import ThresholdSweeper
from sharp.tasks.plot.base import FigureMaker
from sharp.tasks.signal.base import EnvelopeMaker
from sharp.tasks.signal.reference import MakeReference


class PaperGridPlotter(FigureMaker):
    envelope_maker: EnvelopeMaker = TaskParameter()
    reference_makers = [
        MakeReference(**args) for args in config.make_reference_args
    ]
    output_dir = FigureMaker.output_dir / "minipaper"
    cmap = "viridis"

    @property
    def sweepers(self):
        return [
            ThresholdSweeper(
                envelope_maker=self.envelope_maker, reference_maker=rm
            )
            for rm in self.reference_makers
        ]

    def requires(self):
        return self.sweepers

    def output(self):
        return FigureTarget(self.output_dir, self.filename)

    @property
    def filename(self):
        return f"{self.__class__.__name__}-{self.envelope_maker.title}"

    def work(self):
        fig, ax = subplots(figsize=paperfig(0.6, 0.6))
        ax.imshow(
            self.data_matrix,
            cmap=self.cmap,
            origin="lower",
            extent=self.extents,
            aspect="auto",
            vmin=self.cmap_min,
            vmax=self.cmap_max,
        )
        ax.set_xlabel("Min. ripple strength ($\mu$V)")
        ax.set_ylabel("Min. SW strength ($\mu$V)")
        fig.tight_layout()
        self.output().write(fig)

    @property
    def extents(self):
        min_SW = MakeReference(
            mult_detect_SW=config.mult_detect_SW[0],
            mult_detect_ripple=config.mult_detect_ripple[0],
        ).threshold_detect_SW
        max_SW = MakeReference(
            mult_detect_SW=config.mult_detect_SW[-1],
            mult_detect_ripple=config.mult_detect_ripple[0],
        ).threshold_detect_SW
        min_ripple = MakeReference(
            mult_detect_SW=config.mult_detect_SW[0],
            mult_detect_ripple=config.mult_detect_ripple[0],
        ).threshold_detect_ripple
        max_ripple = MakeReference(
            mult_detect_SW=config.mult_detect_SW[0],
            mult_detect_ripple=config.mult_detect_ripple[-1],
        ).threshold_detect_ripple
        # (left, right, bottom, top)
        return (min_ripple, max_ripple, min_SW, max_SW)

    @property
    def data_matrix(self):
        return [
            [
                self.get_data(
                    ThresholdSweeper(
                        envelope_maker=self.envelope_maker,
                        reference_maker=MakeReference(
                            mult_detect_ripple=ripple, mult_detect_SW=SW
                        ),
                    )
                    .output()
                    .read()
                )
                for ripple in config.mult_detect_ripple
            ]
            for SW in config.mult_detect_SW
        ]

    def get_data(self, sweep: ThresholdSweep) -> float:
        ...


class AccuracyGrid(PaperGridPlotter):
    cmap_min = 0.6
    cmap_max = 1

    def get_data(self, sweep):
        return max(sweep.F2)


class LatencyGrid(PaperGridPlotter):
    cmap_min = 0.15
    cmap_max = 0.5
    cmap = "viridis_r"

    def get_data(self, sweep):
        return sweep.at_max_F2().rel_delays_median
