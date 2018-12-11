from scipy.signal import butter

from sharp.data.hardcoded.filters.base import HighpassLowpassCombi


class OurButter(HighpassLowpassCombi):
    title = "Bread-n-butter"

    @property
    def tf_high(self):
        return butter(6, self.normalized_passband[0], "high")

    @property
    def tf_low(self):
        return butter(1, self.normalized_passband[1], "low")
