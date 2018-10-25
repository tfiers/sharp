from typing import Optional

from fklab.segments import Segment
from sharp.config.load import config
from sharp.data.types.signal import Signal
from sharp.data.types.slice import Slice


class DataSplit:

    split_fraction: float

    def __init__(self, signal: Signal, segments: Optional[Segment] = None):
        self._signal_full = signal
        self._segments_full = segments

    @property
    def _left_slice(self) -> Slice:
        return self._get_slice([0, self.split_fraction])

    @property
    def _right_slice(self) -> Slice:
        return self._get_slice([self.split_fraction, 1])

    def _get_slice(self, bounds):
        return Slice(bounds, self._signal_full, self._segments_full)


class TrainTestSplit(DataSplit):
    @property
    def split_fraction(self):
        if config.train_first:
            return config.train_fraction
        else:
            return 1 - config.train_fraction

    @property
    def signal_train(self):
        return self._train_slice.signal

    @property
    def segments_train(self):
        return self._train_slice.segments

    @property
    def time_range_train(self):
        return self._train_slice.time_range

    @property
    def signal_test(self):
        return self._test_slice.signal

    @property
    def segments_test(self):
        return self._test_slice.segments

    @property
    def time_range_test(self):
        return self._test_slice.time_range

    @property
    def _train_slice(self) -> Slice:
        if config.train_first:
            return self._left_slice
        else:
            return self._right_slice

    @property
    def _test_slice(self) -> Slice:
        if config.train_first:
            return self._right_slice
        else:
            return self._left_slice
