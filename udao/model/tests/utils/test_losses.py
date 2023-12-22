import torch as th

from ...utils.losses import WMAPELoss


def test_wmape_loss() -> None:
    y = th.tensor([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
    y_hat = th.tensor([[2, 2, 2, 2, 2], [1, 1, 1, 1, 1]])
    loss = WMAPELoss()

    assert loss(y_hat, y) == 2 / 3
