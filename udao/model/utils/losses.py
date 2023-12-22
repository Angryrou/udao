import torch as th
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .utils import EPS


class WMAPELoss(_Loss):
    def __init__(self) -> None:
        super().__init__(reduction="sum")

    def forward(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:
        return th.abs(input - target).sum() / th.clip(target.sum(), EPS)


class MSLELoss(_Loss):
    def forward(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:
        return F.mse_loss(
            th.log(th.clip(input, min=EPS)),
            th.log(th.clip(target, min=EPS)),
            reduction=self.reduction,
        )


class MAPELoss(_Loss):
    def forward(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:
        return th.mean(th.abs(input - target) / (th.clip(target, EPS)))
