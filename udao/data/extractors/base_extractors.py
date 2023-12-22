from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union

import pandas as pd

from ...data.containers import BaseContainer
from ...data.utils.utils import DatasetType

T = TypeVar("T", bound=BaseContainer)


class TrainedExtractor(ABC, Generic[T]):
    trained: bool = True

    def __init__(self) -> None:
        pass

    @abstractmethod
    def extract_features(self, df: pd.DataFrame, split: DatasetType) -> T:
        pass


class StaticExtractor(ABC, Generic[T]):
    trained: bool = False

    def __init__(self) -> None:
        pass

    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> T:
        pass


FeatureExtractor = Union[TrainedExtractor, StaticExtractor]
