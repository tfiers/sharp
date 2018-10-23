from sharp.data.types.evaluation.sweep import ThresholdSweep
from sharp.data.files.stdlib import PickleFile


class ThresholdSweepFile(PickleFile):
    extension = ".eval" + PickleFile.extension

    def read(self) -> ThresholdSweep:
        return super().read()

    def write(self, sweep: ThresholdSweep):
        super().write(sweep)
