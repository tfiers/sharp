from abc import ABC, abstractmethod

import h5py
import numpy as np
from matplotlib.figure import Figure

import fileflow
from fileflow.file import ContainedDatatype


class HDF5File(fileflow.File, ABC):
    extension = ".hdf5"

    @abstractmethod
    def write_to_file(self, object: ContainedDatatype, f: h5py.File):
        ...

    @abstractmethod
    def read_from_file(self, f: h5py.File) -> ContainedDatatype:
        ...

    def write(self, object: ContainedDatatype):
        # Mode 'w' clears (truncates) the file if it already exists. 'a' leaves
        # existing contents intact.
        with h5py.File(self.path, mode="a") as f:
            f.attrs["Generated with class"] = self.__class__.__qualname__
            f.attrs[".. from Python module"] = __name__
            f.attrs[".. in file"] = __file__
            self.write_to_file(object, f)

    def read(self) -> ContainedDatatype:
        with h5py.File(str(self.path), mode="r") as f:
            object = self.read_from_file(f)
        return object

    @property
    @classmethod
    def main_key(cls):
        """
        A nice default name, for when the file contains one main dataset or
        group.
        """
        return cls.__name__

    @staticmethod
    def read_into_memory(dataset: h5py.Dataset) -> np.ndarray:
        return dataset[()]


class ArrayFile(HDF5File[np.ndarray]):
    def write_to_file(self, array: np.ndarray, f: h5py.File):
        f.create_dataset(self.main_key, data=array)

    def read_from_file(self, f: h5py.File) -> np.ndarray:
        dataset = f[self.main_key]
        array = self.read_into_memory(dataset)
        return array


class FigureFile(fileflow.File[Figure]):
    def write(self, fig: Figure):
        fig.savefig(self.path)
