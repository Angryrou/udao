from typing import Any, Dict, List, Sequence

import torch as th

from ..containers import TabularContainer
from .base_iterator import BaseIterator


class TabularIterator(BaseIterator[th.Tensor, Dict[str, Any]]):
    """Iterator on tabular data.

    Parameters
    ----------
    keys : Sequence[str]
        Keys of the dataset, used for accessing all features
    table : TabularContainer
        Container for the tabular data
    """

    def __init__(
        self,
        keys: Sequence[str],
        tabular_feature: TabularContainer,
    ):
        super().__init__(keys)
        self.tabular_feature = tabular_feature

    def __len__(self) -> int:
        return len(self.keys)

    def _getitem(self, idx: int) -> th.Tensor:
        key = self.keys[idx]
        return th.tensor(self.tabular_feature.get(key), dtype=self.tensors_dtype)

    @property
    def shape(self) -> Any:
        sample_input = self._get_sample()
        return {"input_shape": sample_input.shape}

    @staticmethod
    def collate(items: List[th.Tensor]) -> th.Tensor:
        return th.stack(items, dim=0)
