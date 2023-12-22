from typing import Any

import pytest
import torch as th
from torch import nn
from torchmetrics import WeightedMeanAbsolutePercentageError

from ..module import UdaoModule
from ..utils.losses import WMAPELoss
from ..utils.utils import set_deterministic_torch


@pytest.fixture
def sample_module() -> UdaoModule:
    set_deterministic_torch(0)

    model = nn.Linear(2, 2)
    objectives = ["obj1", "obj2"]
    loss = WMAPELoss()
    module = UdaoModule(
        model=model,
        objectives=objectives,
        loss=loss,
        metrics=[WeightedMeanAbsolutePercentageError],
    )
    return module


class TestUdaoModule:
    def test_initialize(self, sample_module: UdaoModule) -> None:
        assert sample_module.loss_weights == {"obj1": 1.0, "obj2": 1.0}
        assert isinstance(sample_module.optimizer, th.optim.AdamW)
        # Model, loss and metrics should be children of the module
        assert len([child for child in sample_module.children()]) == 3

    def test_compute_loss(self, sample_module: UdaoModule) -> None:
        y = th.tensor([[1, 1], [2, 2]], dtype=th.float32)
        y_hat = th.tensor([[2, 2], [1, 1]], dtype=th.float32)
        loss, loss_dict = sample_module.compute_loss(y, y_hat)
        assert th.allclose(loss, th.tensor(4 / 3))
        assert loss_dict == {"obj1": 2 / 3, "obj2": 2 / 3}

    def test_training_step(self, sample_module: UdaoModule, mocker: Any) -> None:
        batch = (
            th.tensor([[1, 1], [2, 2]], dtype=th.float32),
            th.tensor([[2, 2], [1, 1]], dtype=th.float32),
        )
        mocked_log = mocker.patch.object(sample_module, "log")
        returned = sample_module.training_step(batch, 0)
        assert mocked_log.called
        assert th.allclose(returned, th.tensor(2.7835))

    def test__shared_epoch_end(self, sample_module: UdaoModule, mocker: Any) -> None:
        batch = (
            th.tensor([[1, 1], [2, 2]], dtype=th.float32),
            th.tensor([[2, 2], [1, 1]], dtype=th.float32),
        )

        mocked_log = mocker.spy(sample_module, "log")
        loss = sample_module.training_step(batch, 0)

        sample_module._shared_epoch_end("train")
        # assert the metrics are logged once per objective
        assert mocked_log.call_count == 3
        logged_loss = mocked_log.call_args_list[0][0]
        assert logged_loss[0] == "train_loss"
        obj1_metric = mocked_log.call_args_list[1][0]
        assert obj1_metric[0] == "train_obj1_WeightedMeanAbsolutePercentageError"
        obj2_metric = mocked_log.call_args_list[2][0]
        assert obj2_metric[0] == "train_obj2_WeightedMeanAbsolutePercentageError"

        # assert the loss is the same as the metrics (WMAPE in both cases)
        assert th.equal(obj1_metric[1] + obj2_metric[1], loss)
        assert th.allclose(
            obj1_metric[1],
            th.tensor(0.8075),
            rtol=1e-4,
        )
        assert th.allclose(
            obj2_metric[1],
            th.tensor(1.9760),
            rtol=1e-4,
        )
        assert th.equal(logged_loss[1], loss)

    def test_shared_step(self, sample_module: UdaoModule) -> None:
        batch = (
            th.tensor([[0.5, 2], [0.5, 2]], dtype=th.float32),
            th.tensor([[1, 1], [1, 1]], dtype=th.float32),
        )
        y_hat, y = sample_module._shared_step(batch, "train")
        assert y_hat.shape == (2, 2)
        assert y.shape == (2, 2)
