"""
Describes how raw data is (expected to be) organised, and how it can be read.
"""

import re
from pathlib import Path
from typing import Sequence, Union

from fklab.io.common.binary import _DataProxy_MemoryMap
from fklab.io.neuralynx import NlxFileCSC, NlxOpen
from sharp.data.files.base import InputFileTarget
from sharp.data.files.config import data_config
from sharp.tasks.base import ExternalTask
from sharp.util import cached


class Neuralynx_NCS_File(InputFileTarget):

    extension = ".ncs"

    @property
    def probe_number(self) -> Union[int, None]:
        """ Tetrode (or probe) number. """
        return self._get_number_in_filename(-2)

    @property
    def electrode_number(self) -> Union[int, None]:
        """ Channel of the tetrode / probe. """
        return self._get_number_in_filename(-1)

    def _get_number_in_filename(self, index: int) -> Union[int, None]:
        try:
            return self._numbers_in_filename[index]
        except IndexError:
            return None

    @property
    def _numbers_in_filename(self) -> Sequence[int]:
        """ Integer numbers present in filename, from left to right. """
        filename = self.stem
        numbers = re.findall(r"\d+", filename)
        return [int(number) for number in numbers]

    @property
    @cached
    def _NCS_interface(self) -> NlxFileCSC:
        return NlxOpen(str(self))

    @property
    def fs(self) -> float:
        return self._NCS_interface.header["SamplingFrequency"]

    @property
    def signal_mmap(self) -> _DataProxy_MemoryMap:
        return self._NCS_interface.data.signal_by_sample

    def read(self):
        """
        The file is probably too large to read into memory at once.
        Use `signal_mmap` and `fs` directly.
        """


class Neuralynx_NCS_Directory(ExternalTask):
    """
    Describes a directory that contains Neuralynx continuous recording files.
    """

    def output(self) -> Sequence[Neuralynx_NCS_File]:
        """
        Absolute paths to all Neuralynx *.ncs files in the "data_key"
        directory. The files are ordered first by tetrode (or probe) number,
        and then by channel number.
        """
        paths = self.output_dir.glob("*" + Neuralynx_NCS_File.extension)
        files = [Neuralynx_NCS_File(path.parent, path.stem) for path in paths]
        return sorted(files, key=self.sorting_key)

    @property
    def output_dir(self):
        return Path(data_config.raw_data_dir)

    @staticmethod
    def sorting_key(file: Neuralynx_NCS_File) -> (int, int):
        return (file.probe_number, file.electrode_number)

    @cached
    def get_file(
        self, probe_number: int, electrode_number: int
    ) -> Neuralynx_NCS_File:
        for file in self.output():
            if (file.probe_number == probe_number) and (
                file.electrode_number == electrode_number
            ):
                return file
        else:
            raise FileNotFoundError(
                f"Could not find .ncs file of tetrode (or probe) {probe_number} "
                f"and electrode {electrode_number} in {self.output_dir}."
            )
