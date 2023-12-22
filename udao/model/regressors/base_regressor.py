from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch as th
from torch import nn


class BaseRegressor(nn.Module, ABC):
    @dataclass
    class Params:
        input_embedding_dim: int
        """Size of the embedding part of the input."""
        input_features_dim: int  # depends on the data
        """Size of the tabular features."""
        output_dim: int
        """Size of the output tensor."""

    def __init__(self, net_params: Params) -> None:
        super().__init__()
        self.input_dim = net_params.input_embedding_dim + net_params.input_features_dim
        self.output_dim = net_params.output_dim

    @abstractmethod
    def forward(self, embedding: th.Tensor, inst_feat: th.Tensor) -> th.Tensor:
        ...
