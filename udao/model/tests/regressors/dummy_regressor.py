from dataclasses import dataclass

import torch as th

from ...regressors.base_regressor import BaseRegressor


class DummyRegressor(BaseRegressor):
    @dataclass
    class Params(BaseRegressor.Params):
        pass

    def __init__(self, net_params: Params) -> None:
        super().__init__(net_params)

    def forward(self, embedding: th.Tensor, inst_feat: th.Tensor) -> th.Tensor:
        input_vector = th.cat([embedding, inst_feat], dim=1)
        return th.sum(input_vector, dim=1)
