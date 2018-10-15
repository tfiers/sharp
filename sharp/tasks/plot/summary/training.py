from matplotlib.pyplot import figure

from sharp.data.files.figure import FigureTarget
from sharp.tasks.neuralnet.select import GatherValidLosses
from sharp.tasks.plot.base import FigureMaker


class PlotValidLoss(FigureMaker):
    """
    Plots validation loss versus epoch number for all training epochs.
    """

    valid_loss_gatherer = GatherValidLosses()

    def requires(self):
        return self.valid_loss_gatherer

    def output(self):
        return FigureTarget(self.output_dir, "valid-loss")

    def run(self):
        losses = self.valid_loss_gatherer.output().read()
        fig = figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(losses)
        self.output().write(fig)
