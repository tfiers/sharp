from matplotlib.figure import Figure

from sharp.data.files.base import OutputFileTarget
from sharp.config.load import config


class MatplotlibFigureFile(OutputFileTarget):
    def write(self, fig: Figure):
        fig.savefig(self, bbox_inches="tight")


class BitmapFigureFile(MatplotlibFigureFile):
    extension = ".png"


class PDF_FigureFile(MatplotlibFigureFile):
    extension = ".pdf"


class FigureTarget(OutputFileTarget):
    """
    Saves both bitmap and vector versions of a Matplotlib Figure to disk.
    """

    @property
    def bitmap_version(self):
        return BitmapFigureFile(self.parent / "_png", self.filename)

    @property
    def vector_version(self) -> PDF_FigureFile:
        return PDF_FigureFile(self.parent, self.filename)

    def exists(self):
        if config.bitmap_versions:
            return self.vector_version.exists() and self.bitmap_version.exists()
        else:
            return self.vector_version.exists()

    def write(self, fig: Figure):
        self.vector_version.write(fig)
        if config.bitmap_versions:
            self.bitmap_version.write(fig)

    def delete(self):
        self.vector_version.delete()
        if config.bitmap_versions:
            self.bitmap_version.delete()

    @property
    def filename(self) -> str:
        return self.parts[-1]
