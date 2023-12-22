from dataclasses import dataclass
from typing import List, Optional

import torch as th

from .base_regressor import BaseRegressor
from .layers.mlp_readout import MLPReadout


class MLP(BaseRegressor):
    """MLP to compute the final regression from
    the embedding and the tabular input features.
    Parameters
    ----------
    net_params : MLP.Params
        For the parameters, see the MLP.Params dataclass.
    """

    @dataclass
    class Params(BaseRegressor.Params):
        n_layers: int
        """Number of layers in the MLP"""
        hidden_dim: int
        """Size of the hidden layers outputs."""
        dropout: float
        """Probability of dropout."""
        agg_dims: Optional[List[int]] = None
        """Dimensions of the aggregation layers in the MLP."""

    def __init__(self, net_params: Params) -> None:
        """_summary_"""
        super().__init__(net_params)

        self.MLP_layers = MLPReadout(
            input_dim=self.input_dim,
            hidden_dim=net_params.hidden_dim,
            output_dim=net_params.output_dim,
            n_layers=net_params.n_layers,
            dropout=net_params.dropout,
            agg_dims=net_params.agg_dims,
        )

    def forward(self, embedding: th.Tensor, inst_feat: th.Tensor) -> th.Tensor:
        hgi = th.cat([embedding, inst_feat], dim=1)
        out = self.MLP_layers.forward(hgi)
        return th.exp(out)
