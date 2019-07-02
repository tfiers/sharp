from typing import List, Sequence

from h5py import Dataset, File as HDF5Reader
from numpy import ndarray

from farao import File
from sharp.util.misc import cached


class HDF5File(File):
    extension = ".hdf5"

    def open_file_for_read(self) -> HDF5Reader:
        """ Context manager for reading. """
        return HDF5Reader(self.path_string, "r")

    def open_file_for_write(self, clear_existing: bool = False) -> HDF5Reader:
        """ Context manager for writing. """
        if clear_existing:
            mode = "w"
        else:
            mode = "a"
        return HDF5Reader(self.path_string, mode)


class ArrayFile(HDF5File):
    KEY = "array"

    def read(self) -> ndarray:
        with self.open_file_for_read() as f:
            data = f[self.KEY][()]
        return data

    def write(self, array: ndarray):
        with self.open_file_for_write() as f:
            f.create_dataset(self.KEY, data=array)


class ArrayListFile(HDF5File):
    """
    A list of arbitrarily sized and shaped NumPy arrays, stored on disk.
    """

    @cached
    def read(self) -> List[ndarray]:
        with self.open_file_for_read() as f:
            datasets: List[Dataset] = f.values()
            array_list = [None] * len(datasets)
            for dataset in datasets:
                full_path: str = dataset.name
                name = full_path.split("/")[-1]
                channel_nr = int(name)
                array_list[channel_nr] = dataset[()]
        return array_list

    def write(self, arrays: Sequence[ndarray]):
        with self.open_file_for_write() as f:
            for channel_nr, array in enumerate(arrays):
                f.create_dataset(name=str(channel_nr), data=array)
