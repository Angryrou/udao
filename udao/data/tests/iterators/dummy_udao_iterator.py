from typing import Sequence, Tuple

import torch as th

from ....data.containers.tabular_container import TabularContainer
from ....data.iterators.base_iterator import UdaoIterator
from ....utils.interfaces import (
    UdaoEmbedInput,
    UdaoEmbedItemShape,
    UdaoInput,
    UdaoItemShape,
)


class DummyUdaoIterator(UdaoIterator[UdaoInput, UdaoItemShape]):
    def __init__(
        self,
        keys: Sequence[str],
        tabular_features: TabularContainer,
        objectives: TabularContainer,
    ) -> None:
        super().__init__(keys, tabular_features=tabular_features, objectives=objectives)

    def _getitem(self, idx: int) -> Tuple[UdaoInput, th.Tensor]:
        key = self.keys[idx]
        return (
            UdaoInput(
                th.tensor(self.tabular_features.get(key), dtype=self.tensors_dtype)
            ),
            th.tensor(self.objectives.get(key), dtype=self.tensors_dtype),
        )

    @property
    def shape(self) -> UdaoItemShape:
        return UdaoItemShape(
            feature_names=list(self.tabular_features.data.columns),
            output_names=list(self.objectives.data.columns),
        )

    @staticmethod
    def collate(
        items: Sequence[Tuple[UdaoInput, th.Tensor]]
    ) -> Tuple[UdaoInput, th.Tensor]:
        features = UdaoInput(th.vstack([item[0].features for item in items]))
        objectives = th.vstack([item[1] for item in items])
        return features, objectives


class DummyUdaoEmbedIterator(UdaoIterator[UdaoEmbedInput, UdaoEmbedItemShape]):
    def __init__(
        self,
        keys: Sequence[str],
        tabular_features: TabularContainer,
        objectives: TabularContainer,
        embedding_features: TabularContainer,
    ) -> None:
        self.embedding_features = embedding_features
        super().__init__(keys, tabular_features=tabular_features, objectives=objectives)

    def _getitem(self, idx: int) -> Tuple[UdaoEmbedInput, th.Tensor]:
        key = self.keys[idx]
        return (
            UdaoEmbedInput(
                embedding_input=th.tensor(
                    self.embedding_features.get(key), dtype=self.tensors_dtype
                ),
                features=th.tensor(
                    self.tabular_features.get(key), dtype=self.tensors_dtype
                ),
            ),
            th.tensor(self.objectives.get(key), dtype=self.tensors_dtype),
        )

    @property
    def shape(self) -> UdaoEmbedItemShape:
        return UdaoEmbedItemShape(
            embedding_input_shape=self.embedding_features.data.shape[1],
            feature_names=list(self.tabular_features.data.columns),
            output_names=list(self.objectives.data.columns),
        )

    @staticmethod
    def collate(
        items: Sequence[Tuple[UdaoEmbedInput, th.Tensor]]
    ) -> Tuple[UdaoEmbedInput, th.Tensor]:
        embedding_input = th.vstack([item[0].embedding_input for item in items])
        features = th.vstack([item[0].features for item in items])
        objectives = th.vstack([item[1] for item in items])
        return (
            UdaoEmbedInput(embedding_input=embedding_input, features=features),
            objectives,
        )
