import numpy as np
import torch
from numpy.testing import assert_equal

from sharp.data.types.neuralnet import RNN
from sharp.tasks.signal.util import view


def test_view():
    # fmt: off
    assert_equal(view(1, 0.1),
                 np.array([0.9, 1.1]))

    assert_equal(view([1], 0.1),
                 np.array([[0.9, 1.1]]))

    assert_equal(view([1, 2], 0.1),
                 np.array([[0.9, 1.1],
                           [1.9, 2.1]]))

    assert_equal(view([1, 2, 3], 0.1),
                 np.array([[0.9, 1.1],
                           [1.9, 2.1],
                           [2.9, 3.1]]))
    # fmt: on


def test_pytorch():
    net = RNN(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    h0 = net.get_init_h()
    i = h0.new_zeros((1, 300, 1))
    out, hn = net(i, h0)
