from farao import File
from h5py import File as HDF5Reader


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
