from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np


class BasePredicateEmbedder(ABC):
    @abstractmethod
    def fit_transform(
        self, training_texts: Sequence[str], epochs: Optional[int] = None
    ) -> np.ndarray:
        pass

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> np.ndarray:
        pass
