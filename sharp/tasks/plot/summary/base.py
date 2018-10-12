from sharp.tasks.plot.base import PosterFigureMaker


class SummaryFigureMaker(PosterFigureMaker):
    @property
    def output_dir(self):
        return super().output_dir / "summary"
