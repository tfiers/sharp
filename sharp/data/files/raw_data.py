from abc import ABC, abstractmethod
from logging import getLogger
from typing import TypeVar

from numpy import int16, memmap

from fklab.io.common import BinaryFileReader
from fklab.io.neuralynx import NlxOpen
from sharp.data.files.base import HDF5Target, InputFileTarget
from sharp.util.misc import cached

log = getLogger(__name__)

OpenedFile = TypeVar("OpenedFile")


class RawRecording(InputFileTarget, ABC):
    def read(self):
        return self.signal

    def close(self):
        """ Please call me when ready, to avoid unnecessary file locks. """
        del self.__opened_file

    __opened_file: OpenedFile = None

    @property
    def opened_file(self) -> OpenedFile:
        if self.__opened_file is None:
            self.__opened_file = self._open()
        return self.__opened_file

    @abstractmethod
    def _open(self) -> OpenedFile:
        """ Return a file-like object. """

    @property
    @abstractmethod
    def signal(self):
        """
        Returns an object that conforms to the `signal` argument of
        `downsample_chunkwise` from `fklab.signal.multirate`.
        """

    @property
    @abstractmethod
    def fs(self) -> float:
        """ Sampling frequency of `signal`, in Hz. """

    @property
    @abstractmethod
    def to_microvolts(self) -> float:
        """
        Multiplicative conversion factor to convert the data values in `signal`
        to microvolts.
        """


class TahitiFile(RawRecording):
    extension = ".moz"

    def _open(self) -> BinaryFileReader:
        return NlxOpen(self)

    @property
    def signal(self):
        # fklab's BinaryFileReader creates NumPy memmap's internally. The disk
        # file opened can be closed by deleting the NumPy memmap objects.
        return self.opened_file.data.data_by_sample

    @property
    @cached
    def fs(self):
        return self.opened_file.header["SamplingFrequency"]

    to_microvolts = 1


class RawKwikFile(RawRecording, HDF5Target):
    """
    A custom HDF5 file created by the KlustaKwik spike sorting software.
    Specification: https://github.com/klusta-team/kwiklib/wiki/Kwik-format
    """

    extension = ".raw.kwd"

    def _open(self):
        return self.open_file_for_read()

    def close(self):
        self.opened_file.close()
        super().close()

    @property
    def signal(self):
        return self.opened_file["recordings/0/data"]

    @property
    @cached
    def fs(self):
        return self.get_attribute("channel_sample_rates")

    @property
    @cached
    def to_microvolts(self):
        # Name of attribute is misleading, as its value (0.195) is microvolts
        # per (int16) bit, not volts per bit. See Intan headstage datasheet:
        # http://intantech.com/files/Intan_RHD2000_series_datasheet.pdf
        # (column 'units' has uV).
        return self.get_attribute("channel_bit_volts")

    def get_attribute(self, name: str) -> float:
        info_path = "recordings/0/application_data"
        values = self.opened_file[info_path].attrs[name]
        if not all(values == values[0]):
            log.warning(
                f"Not all channels have the same \"{name}\" value in file {self}."
            )
        return values[0]


class RawDATFile(RawRecording):
    extension = ".dat"
    fs = 32000  # Taken from the corresponding .prm file.
    to_microvolts = 0.183_111_06
    # Based on a manual comparison with one of the .ncs files corresponding to
    # a .dat file.

    NUM_CHANNELS = 16

    def _open(self) -> memmap:
        return memmap(self.path_string, dtype=int16).reshape(
            (-1, self.NUM_CHANNELS)
        )

    @property
    def signal(self):
        return self.opened_file
