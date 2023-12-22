from typing import Dict, List

import torch as th
from torchmetrics import Metric

from .utils import EPS


class ErrorPercentiles(Metric):
    def __init__(
        self,
        percentiles: List[float] = [50, 90, 95, 99],
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.quantiles = th.tensor(percentiles) / 100
        self.add_state("error_rate", default=th.empty((0)), dist_reduce_fx=None)

    def update(self, input: th.Tensor, target: th.Tensor) -> None:
        new_error_rate = th.abs(input - target) / (th.clip(target, min=EPS))
        self.error_rate = th.cat([self.error_rate, new_error_rate])  # type: ignore

    def compute(self) -> Dict[str, th.Tensor]:
        error_quantiles = th.quantile(self.error_rate, self.quantiles, dim=0)  # type: ignore
        return {
            "mean": self.error_rate.mean(),  # type: ignore
            **{
                f"q_{int(q * 100)}": error_quantiles[i]
                for i, q in enumerate(self.quantiles)
            },
        }


class QErrorPercentiles(Metric):
    def __init__(
        self,
        percentiles: List[int] = [50, 90, 95, 99],
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.quantiles = th.tensor(percentiles) / 100
        self.add_state("q_error", default=th.empty((0)), dist_reduce_fx=None)

    def update(self, input: th.Tensor, target: th.Tensor) -> None:
        new_q_error = th.maximum(
            th.clip(target, EPS) / th.clip(input, EPS),
            th.clip(input, EPS) / th.clip(target, EPS),
        )
        self.q_error = th.cat([self.q_error, new_q_error])  # type: ignore

    def compute(self) -> Dict[str, th.Tensor]:
        q_error_quantiles = th.quantile(self.q_error, self.quantiles, dim=0)  # type: ignore
        return {
            "mean": self.q_error.mean(),  # type: ignore
            **{
                f"q_{int(q * 100)}": q_error_quantiles[i]
                for i, q in enumerate(self.quantiles)
            },
        }
