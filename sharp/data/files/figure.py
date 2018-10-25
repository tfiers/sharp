from matplotlib.figure import Figure

from sharp.data.files.base import OutputFileTarget
from sharp.config.params import data_config


class MatplotlibFigureFile(OutputFileTarget):
    def write(self, fig: Figure):
        fig.savefig(self)


class BitmapFigureFile(MatplotlibFigureFile):
    extension = ".png"


class VectorFigureFile(MatplotlibFigureFile):
    extension = ".pdf"


class FigureTarget(OutputFileTarget):
    """
    Saves both bitmap and vector versions of a Matplotlib Figure to disk.
    """

    def exists(self):
        if data_config.bitmap_versions:
            return self.vector_version.exists() and self.bitmap_version.exists()
        else:
            return self.vector_version.exists()

    def write(self, fig: Figure):
        self.vector_version.write(fig)
        if data_config.bitmap_versions:
            self.bitmap_version.write(fig)

    def delete(self):
        self.vector_version.delete()
        if data_config.bitmap_versions:
            self.bitmap_version.delete()

    @property
    def bitmap_version(self):
        return BitmapFigureFile(self.parent / "_png", self.filename)

    @property
    def vector_version(self) -> VectorFigureFile:
        return VectorFigureFile(self.parent, self.filename)

    @property
    def filename(self) -> str:
        return self.parts[-1]
