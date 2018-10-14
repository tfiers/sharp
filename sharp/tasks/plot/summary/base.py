from sharp.tasks.plot.base import MultiEnvelopeFigureMaker


class SummaryFigureMaker(MultiEnvelopeFigureMaker):
    @property
    def output_dir(self):
        return super().output_dir / "summaries"
