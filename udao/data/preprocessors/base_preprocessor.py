from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union

from ..containers.base_container import BaseContainer
from ..utils.utils import DatasetType

T = TypeVar("T", bound=BaseContainer)


class TrainedPreprocessor(ABC, Generic[T]):
    """Base class for feature processors that require training."""

    trained: bool = True

    def __init__(self) -> None:
        pass

    @abstractmethod
    def preprocess(self, container: T, split: DatasetType) -> T:
        pass


class StaticPreprocessor(ABC, Generic[T]):
    """Base class for feature processors that do not require training."""

    trained: bool = False

    def __init__(self) -> None:
        pass

    @abstractmethod
    def preprocess(self, container: T) -> T:
        pass


FeaturePreprocessor = Union[TrainedPreprocessor, StaticPreprocessor]
