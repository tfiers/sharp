from matplotlib import style

from sharp.data.files.config import output_root
from sharp.tasks.base import SharpTask
from sharp.tasks.evaluate.algorithms import EvaluateAlgorithms
from sharp.tasks.plot.style import symposium


class FigureMaker(SharpTask):
    @property
    def output_dir(self):
        return output_root / "figures"

    def __init__(self, *args, **kwargs):
        style.use(symposium)
        super().__init__(*args, **kwargs)


class PosterFigureMaker(FigureMaker):
    @property
    def output_dir(self):
        return super().output_dir / "poster-Cell-NERF"

    evaluation = EvaluateAlgorithms()

    def requires(self):
        return self.evaluation

    # fmt:off
    @property
    def titles(self):
        return self.evaluation.sort_values_by_sweeper({
            "sota": "Band-pass filter",
            "proposal": "Recurrent neural net",
        })

    @property
    def colors(self):
        return self.evaluation.sort_values_by_sweeper({
            "sota": "C0",
            "proposal": "C1"
        })
    # fmt: on
