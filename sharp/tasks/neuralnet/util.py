import torch

from sharp.data.types.aliases import TorchArray, NumpyArray


def numpy_to_torch(array: NumpyArray) -> TorchArray:
    """
    Convert a NumPy array to a PyTorch tensor, of the right data type. The
    input and output objects share the same data in memory.
    """
    return torch.from_numpy(array).float()


def to_batch(array: TorchArray, one_sample: bool = True) -> TorchArray:
    """
    PyTorch functions take data in "batches", i.e. with multiple "samples" at
    once. This function adds a size-one 'batch' dimension to the given array,
    if necessary.

    one_sample: whether the given array consists of a single (training) example.

    For RNN's e.g., one "sample" is one (single- or multichannel) signal
    excerpt.
    """
    if one_sample:
        # The first dimension is (most often) the batch dimension.
        return array.unsqueeze(0)
    else:
        if array.ndimension() == 1:
            # The array is a vector, with each element a different sample.
            return array.unsqueeze(1)
        else:
            return array
