from luigi import IntParameter, Parameter
from numpy.core.multiarray import arange

from sharp.config.channels import CHANNEL_COMBINATIONS


class GEVecMixin:

    num_delays = IntParameter()
    # Set to zero to use only the current sample.
    # (number of temporal samples used = num_delays + 1)

    channel_combo_name = Parameter(default="all")

    @property
    def delays(self):
        """ Includes the current time sample, as delay = 0. """
        return arange(self.num_delays + 1)

    @property
    def channels(self):
        return CHANNEL_COMBINATIONS[self.channel_combo_name]

    @property
    def num_channels(self):
        return len(self.channels)

    @property
    def _num_delays_str(self) -> str:
        if self.num_delays == 0:
            return "no delays"
        elif self.num_delays == 1:
            return "1 delay"
        else:
            return f"{self.num_delays} delays"

    @property
    def _channels_str(self):
        if self.num_channels == 1:
            return f"{self.channel_combo_name} channel"
        else:
            return f"{self.channel_combo_name} channels"

    @property
    def filename(self):
        return f"{self._num_delays_str}, {self._channels_str}"
