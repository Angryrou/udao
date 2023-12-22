from abc import abstractmethod
from inspect import signature
from typing import Any, Callable, Generic, List, Sequence, Tuple, TypeVar

import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Dataset

from ...utils.interfaces import UdaoInput, UdaoItemShape
from ..containers import TabularContainer

T = TypeVar("T")
ST = TypeVar("ST")


class BaseIterator(Dataset, Generic[T, ST]):
    """Base class for all dataset iterators.
    Inherits from torch.utils.data.Dataset.

    T is the type of the iterator output.
    ST is the type of the iterator output shape.

    See FeatureIterator for an example.
    """

    def __init__(self, keys: Sequence[str]) -> None:
        self.keys = keys
        self.tensors_dtype = th.float32
        self.augmentations: List[Callable[[T], T]] = []
        pass

    def __len__(self) -> int:
        return len(self.keys)

    @abstractmethod
    def _getitem(self, idx: int, /) -> T:
        pass

    def __getitem__(self, idx: int, /) -> T:
        """Returns the item at the given index."""
        item = self._getitem(idx)
        for augmentation in self.augmentations:
            item = augmentation(item)
        return item

    def set_augmentations(self, augmentations: List[Callable[[T], T]]) -> None:
        """Sets the augmentations to apply to the iterator output."""
        self.augmentations = augmentations

    @property
    @abstractmethod
    def shape(self) -> ST:
        """Returns the shape of the iterator output."""

    @staticmethod
    @abstractmethod
    def collate(items: List[T]) -> T:
        """Collates the items into a batch.
        Used in the dataloader."""
        pass

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> DataLoader:
        """Returns a torch dataloader for the iterator,
        that can be used for training.
        This will use the collate static method
        to collate the items into a batch.
        """
        return DataLoader[T](
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def get_parameter_names(cls) -> List[str]:
        """Returns the names of the container parameters
        of the iterator.
        Useful to create dynamic parameters for related parts
        of the pipeline (feature extractors, preprocessors)
        """
        init_signature = signature(cls.__init__)
        params_list = list(init_signature.parameters.keys())
        # remove self, args and kwargs
        params_list = [
            param
            for param in params_list
            if param not in ["self", "keys", "args", "kwargs"]
        ]
        return params_list

    def _get_sample(self) -> T:
        """Returns a random sample from the iterator."""
        return self[0]

    def set_tensors_dtype(self, dtype: th.dtype) -> None:
        """Sets the dtype of the iterator.
        Useful for mixed precision training.
        """
        self.tensors_dtype = dtype


# Type of the iterator output - in the Udao case,
# restricted to UdaoInput and its subclasses
UT = TypeVar("UT", bound=UdaoInput)
# Type of the iterator output shape - in the Udao case,
# restricted to UdaoInputShape and its subclasses
UST = TypeVar("UST", bound=UdaoItemShape)


class UdaoIterator(BaseIterator[Tuple[UT, th.Tensor], UST], Generic[UT, UST]):
    """Base iterator for the Udao use case, where the iterator
    returns a FeatureInput object. It is expected to accept:
    - a TabularContainer representing the tabular features
    which can be set as variables by the user in the optimization pipeline
    - a TabularContainer representing the objectives

    FST: Type of the iterator output shape - in the Udao case,
    restricted to FeatureInputShape and its subclasses.

    FT: Type of the iterator output - in the Udao case,
    restricted to th.Tensor and its subclasses
    This results in a type Tuple[UT, th.Tensor] for the iterator output.

    Parameters
    ----------
    keys : Sequence[str]
        Keys of the dataset, used for accessing all features
    tabular_features : TabularContainer
        Tabular features of the iterator
    objectives : TabularContainer
        Objectives of the iterator
    """

    def __init__(
        self,
        keys: Sequence[str],
        tabular_features: TabularContainer,
        objectives: TabularContainer,
    ) -> None:
        super().__init__(keys)
        self.tabular_features = tabular_features
        self.objectives = objectives

    def get_tabular_features_container(
        self, feature_input: th.Tensor
    ) -> TabularContainer:
        indices = [
            i
            for i, name in enumerate(self.shape.feature_names)
            if name in self.tabular_features.data.columns
        ]
        tabular_features = feature_input[:, indices]
        tabular_df = pd.DataFrame(
            tabular_features.cpu().numpy(), columns=self.tabular_features.data.columns
        )
        return TabularContainer(tabular_df)
