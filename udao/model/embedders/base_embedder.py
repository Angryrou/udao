from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from torch import nn

from udao.utils.interfaces import UdaoEmbedItemShape


class BaseEmbedder(nn.Module, ABC):
    """Placeholder Base class for Embedder networks."""

    @dataclass
    class Params:
        output_size: int
        """The size of the output embedding."""

    @classmethod
    @abstractmethod
    def from_iterator_shape(
        cls,
        iterator_shape: UdaoEmbedItemShape,
        **kwargs: Any,
    ) -> "BaseEmbedder":
        ...

    def __init__(self, net_params: Params) -> None:
        super().__init__()
        self.embedding_size = net_params.output_size
