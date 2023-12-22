import copy
from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar("T", bound="BaseContainer")


class BaseContainer(ABC):
    """Base class for containers.
    Containers are used to store and retrieve data
    from a dataset, based on a key."""

    @abstractmethod
    def get(self, key: str):  # type: ignore
        pass

    def copy(self: T) -> T:
        return copy.deepcopy(self)
