from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

import torch as th

T = TypeVar("T")

ST = TypeVar("ST")


@dataclass
class UdaoInput:
    features: th.Tensor

    def to(self, device: th.device) -> "UdaoInput":
        return UdaoInput(self.features.to(device))


@dataclass
class UdaoItemShape:
    feature_names: list[str]
    output_names: list[str]


@dataclass
class UdaoEmbedInput(Generic[T], UdaoInput):
    embedding_input: T

    def to(self, device: th.device) -> "UdaoEmbedInput":
        if hasattr(self.embedding_input, "to"):
            return UdaoEmbedInput(
                self.embedding_input.to(device), self.features.to(device)  # type: ignore
            )
        else:
            return UdaoEmbedInput(
                self.embedding_input, self.features.to(device)  # type: ignore
            )


@dataclass
class UdaoEmbedItemShape(Generic[ST], UdaoItemShape):
    embedding_input_shape: ST


class VarTypes(Enum):
    INT = "int"
    BOOL = "bool"
    CATEGORY = "category"
    FLOAT = "float"
