from luigi import IntParameter
from numpy import cov
from numpy.core.multiarray import concatenate
from numpy.core.records import ndarray
from scipy.linalg import eig

from sharp.data.files.config import output_root
from sharp.data.files.numpy import NumpyArrayFile
from sharp.tasks.base import SharpTask
from sharp.tasks.signal.base import InputDataMixin


class MaximiseSNR(SharpTask, InputDataMixin):

    num_delays = IntParameter()

    def requires(self):
        return self.input_data_makers

    output_dir = output_root / "linear-filters"

    def output(self):
        return NumpyArrayFile(self.output_dir, "GEVec")

    def run(self):
        signal = self.input_signal_train.as_matrix()
        segments = self.reference_segs_train
        reference = concatenate(signal.extract(segments))
        background = concatenate(signal.extract(segments.invert()))
        Rss = as_matrix(cov(reference, rowvar=False))
        Rnn = as_matrix(cov(background, rowvar=False))
        _, GEVecs = eig(Rss, Rnn)
        first_GEVec = GEVecs[0, :]
        self.output().write(first_GEVec)


def as_matrix(array: ndarray):
    if array.shape == ():
        return array.reshape((1, 1))
    else:
        return array
