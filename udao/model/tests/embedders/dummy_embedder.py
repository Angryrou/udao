from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from ....utils.interfaces import UdaoEmbedItemShape
from ...embedders.base_embedder import BaseEmbedder


class DummyEmbedder(BaseEmbedder):
    @dataclass
    class Params(BaseEmbedder.Params):
        input_size: int

    def __init__(self, params: Params) -> None:
        super().__init__(net_params=params)
        self.input_size = params.input_size
        self.fc_layer = nn.Linear(self.input_size, self.embedding_size)

    @classmethod
    def from_iterator_shape(
        cls,
        iterator_shape: UdaoEmbedItemShape,
        **kwargs: Any,
    ) -> "DummyEmbedder":
        return DummyEmbedder(
            params=DummyEmbedder.Params(
                input_size=iterator_shape.embedding_input_shape, **kwargs
            )
        )

    def forward(self, input: Any) -> Any:
        return input
