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
