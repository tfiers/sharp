import numpy as np
import torch


# Memory location of PyTorch arrays (either at a GPU, or at the CPU).
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np_to_torch(array: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy array to a PyTorch tensor, of the correct data type. The
    input and output objects share the same data in memory.
    """
    # The explicit float (float32) conversion is a gotcha.
    tensor: torch.Tensor = torch.from_numpy(array).float()
    return tensor.to(torch_device)


def to_batch(array: torch.Tensor, one_sample: bool = True) -> torch.Tensor:
    """
    PyTorch functions take data in "batches", i.e. with multiple "samples" at
    once. This function adds a size-one 'batch' dimension to the given array,
    if necessary.

    :param array
    :param one_sample:  Whether the given array consists of a single (training)
            example. For RNN's e.g., one "sample" is one (single- or
            multichannel) signal excerpt.
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
