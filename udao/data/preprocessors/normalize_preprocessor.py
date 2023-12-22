from typing import Any, List, Protocol, TypeVar

import pandas as pd

from ..containers.base_container import BaseContainer
from ..utils.utils import DatasetType
from .base_preprocessor import TrainedPreprocessor


# Define a protocol for objects that have fit and transform methods
class FitTransformProtocol(Protocol):
    def fit(self, X: Any, y: Any = None) -> "FitTransformProtocol":
        ...

    def transform(self, X: Any) -> Any:
        ...

    def inverse_transform(self, X: Any) -> Any:
        ...


T = TypeVar("T", bound=BaseContainer)


class NormalizePreprocessor(TrainedPreprocessor[T]):
    """Normalize the data using a normalizer that
    implements the fit and transform methods, e.g. MinMaxScaler.

    Parameters
    ----------
    normalizer : FitTransformProtocol
        A normalizer that implements the fit and transform methods
        (e.g. sklearn.MinMaxScaler)
    df_key : str
        The key of the dataframe in the container.
    """

    def __init__(
        self, normalizer: FitTransformProtocol, data_key: str = "data"
    ) -> None:
        self.normalizer = normalizer
        self.df_key = data_key
        self.numeric_columns: List[str] = []

    def preprocess(self, container: T, split: DatasetType) -> T:
        """Normalize the data in the container.

        Parameters
        ----------
        container : T
            Child of BaseContainer
        split : DatasetType
            Train or other (val, test).
        Returns
        -------
        T
            Child of BaseContainer with the normalized data.
        """
        container = container.copy()
        df: pd.DataFrame = container.__getattribute__(self.df_key)
        numeric_df = df.select_dtypes(include=["number"])
        self.numeric_columns = numeric_df.columns.tolist()
        if split == "train":
            self.normalizer.fit(numeric_df)

        transformed_data = self.normalizer.transform(numeric_df)
        transformed_df = pd.DataFrame(
            transformed_data, index=numeric_df.index, columns=self.numeric_columns
        )

        df.update(transformed_df)

        container.__setattr__(self.df_key, df)
        return container

    def inverse_transform(self, container: T) -> T:
        """Reverse the normalization process on the container's data.

        Parameters
        ----------
        container : T
            Child of BaseContainer with the normalized data.

        Returns
        -------
        T
            Child of BaseContainer with the data in original scale.
        """
        container = container.copy()
        df: pd.DataFrame = container.__getattribute__(self.df_key)
        numeric_df = df[self.numeric_columns]

        original_data = self.normalizer.inverse_transform(numeric_df.values)
        original_numeric_df = pd.DataFrame(
            original_data, index=numeric_df.index, columns=self.numeric_columns
        )

        df.update(original_numeric_df)

        container.__setattr__(self.df_key, df)
        return container
