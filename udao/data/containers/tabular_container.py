from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base_container import BaseContainer


@dataclass
class TabularContainer(BaseContainer):
    """Container for tabular data, stored in DataFrame format."""

    data: pd.DataFrame

    def get(self, key: str) -> np.ndarray:
        return self.data.loc[key].values  # type: ignore
