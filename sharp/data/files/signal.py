from dataclasses import dataclass

from sharp.data.files.base import HDF5File
from sharp.data.types.signal import Signal
from sharp.util.misc import cached


class SignalFile(HDF5File):
    """
    A `Signal` (subclass of a NumPy array), stored on disk.
    """

    @dataclass
    class Keys:
        SIGNAL = "signal"
        FS = "sampling frequency (Hz)"
        UNITS = "units"

    @cached
    def read(self) -> Signal:
        with self.open_file_for_read() as f:
            dataset = f[self.Keys.SIGNAL]
            array = dataset[:]
            fs = dataset.attrs[self.Keys.FS]
            units = dataset.attrs.get(self.Keys.UNITS, None)
        return Signal(array, fs, units)

    def write(self, signal: Signal):
        with self.open_file_for_write() as f:
            dataset = f.create_dataset(self.Keys.SIGNAL, data=signal.data)
            dataset.attrs[self.Keys.FS] = signal.fs
            if signal.units:
                dataset.attrs[self.Keys.UNITS] = signal.units
