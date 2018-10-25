from luigi import IntParameter, Parameter
from numpy.core.multiarray import arange


class GEVecMixin:

    num_delays = IntParameter()
    # Set to zero to use only the current sample.
    # (number of temporal samples used = num_delays + 1)

    channels = Parameter(default="all")

    @property
    def delays(self):
        """ Includes the current time sample, as delay = 0. """
        return arange(self.num_delays + 1)

    @property
    def num_delays_str(self) -> str:
        n = self.num_delays
        if n == 0:
            return "no delays"
        elif n == 1:
            return "1 delay"
        else:
            return f"{n} delays"

    @property
    def filename(self):
        return self.num_delays_str.replace(" ", "-")
