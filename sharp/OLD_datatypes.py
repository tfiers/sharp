from abc import ABC
from typing import List

import h5py
import numpy as np

import fklab.segments
from farao import Saveable


class SavedAsHDF5(Saveable, ABC):
    extension = ".hdf5"

    @staticmethod
    def file_reader(path) -> h5py.File:
        """ Context manager for reading. """
        return h5py.File(path, mode="r")

    @staticmethod
    def file_writer(path) -> h5py.File:
        """ Context manager for writing. """
        # Mode 'w' clears (truncates) the file if it already exists. 'a' leaves
        # existing contents intact.
        return h5py.File(path, mode="a")


class ArrayList(List[np.ndarray], SavedAsHDF5):
    """
    An ordered collection of arbitrarily sized and shaped NumPy arrays, saved
    as an HDF5 file.
    """

    def save(self, path):
        with self.file_writer() as f:
            for index, array in enumerate(self):
                f.create_dataset(name=str(index), data=array)

    @classmethod
    def load(cls, path) -> "ArrayList":
        with cls.file_reader(path) as f:
            datasets: List[h5py.Dataset] = f.values()
            array_list = [None] * len(datasets)
            for dataset in datasets:
                full_path: str = dataset.name
                name = full_path.split("/")[-1]
                index = int(name)
                array_list[index] = dataset[()]
        return array_list


class Segment(fklab.segments.Segment, SavedAsHDF5):
    # Note: we shouldn't subclass Array (and thus ndarray), as fkklab's Segment
    # is not a subclass of ndarray (Liskov).

    def save(self, path):
        data = self.asarray()
        Array(data).save(path)

    @classmethod
    def load(cls, path):
        return cls(data=ArrayFile.load(path), check=False)


class SegmentList(List[fklab.segments.Segment], SavedAsHDF5):
    """
    A sequence of fklab Segments, stored on disk.
    (Where each "Segment" is actually a full sequence of (start, stop) tuples).
    """

    def save(self, path):
        data = [seg.asarray() for seg in self]
        ArrayList(data).save(path)

    @classmethod
    def load(cls, path):
        arrays = ArrayList.load(path)
        return cls([Segment(data=array, check=False) for array in arrays])
