import torch as th

from ...utils.metrics import ErrorPercentiles, QErrorPercentiles


class TestErrorPercentile:
    def test_compute(self) -> None:
        input_tensor = th.tensor([1, 2.2, 3.3])
        target_tensor = th.tensor([1, 2, 3])
        metric = ErrorPercentiles(percentiles=[50])
        metric.update(input_tensor, target_tensor)
        result = metric.compute()
        assert th.allclose(result["mean"], th.tensor(0.2 / 3))
        assert th.allclose(result["q_50"], th.tensor(0.1))
        metric.reset()
        assert metric.error_rate.shape == (0,)


class TestQErrorPercentile:
    def test_compute(self) -> None:
        input_tensor = th.tensor([1, 2.2, 3.3])
        target_tensor = th.tensor([1, 2, 3])
        metric = QErrorPercentiles(percentiles=[50])
        metric.update(input_tensor, target_tensor)
        result = metric.compute()
        assert th.allclose(result["mean"], th.tensor(1 + 0.2 / 3))
        assert th.allclose(result["q_50"], th.tensor(1.1))
        metric.reset()
        assert metric.q_error.shape == (0,)
