from typing import List, Optional, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ..utils.utils import DatasetType
from .base_preprocessor import T, TrainedPreprocessor


class OneHotPreprocessor(TrainedPreprocessor):
    """One-hot encode the specified categorical features.

    Parameters
    ----------
    encoder : OneHotEncoder
        An instance of sklearn's OneHotEncoder
    categorical_features : List[str]
        The list of column names to encode.
    """

    def __init__(
        self,
        categorical_features: Optional[List[str]],
        data_key: str = "data",
    ) -> None:
        self.encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )
        self.categorical_features = categorical_features
        self.df_key = data_key
        self.encoded_feature_names: Optional[np.ndarray] = None

    def preprocess(self, container: T, split: DatasetType) -> T:
        """Apply one-hot encoding to the data in the container.

        Parameters
        ----------
        container : T
            Child of BaseContainer
        split : DatasetType
            Train or other (val, test).

        Returns
        -------
        T
            Child of BaseContainer with one-hot encoded data.
        """
        container = container.copy()
        df: pd.DataFrame = container.__getattribute__(self.df_key)

        if split == "train":
            if self.categorical_features is None:
                self.categorical_features = df.select_dtypes(
                    include=["category", "object"]
                ).columns.tolist()
            self.encoder.fit(df[self.categorical_features].to_numpy())
            self.encoded_feature_names = self.encoder.get_feature_names_out(
                self.categorical_features
            )

        if self.categorical_features is None:
            raise ValueError(
                "Categorical features not found. "
                "Please preprocess the data before preprocessing."
            )
        transformed_data = cast(
            np.ndarray, self.encoder.transform(df[self.categorical_features])
        )

        encoded_df = pd.DataFrame(
            data=transformed_data, index=df.index, columns=self.encoded_feature_names
        )

        container.__setattr__(
            self.df_key,
            pd.concat([df.drop(self.categorical_features, axis=1), encoded_df], axis=1),
        )
        return container

    def inverse_preprocess(self, container: T) -> T:
        """Reverse the one-hot encoding process on the container's data.

        Parameters
        ----------
        container : T
            Child of BaseContainer with one-hot encoded data.

        Returns
        -------
        T
            Child of BaseContainer with the original categorical data.
        """
        container = container.copy()
        df: pd.DataFrame = container.__getattribute__(self.df_key)
        if self.encoded_feature_names is None:
            raise ValueError(
                "Encoded feature names not found. "
                "Please preprocess the data before inverse preprocessing."
            )
        encoded_columns = [
            col
            for col in df.columns
            if col.startswith(tuple(self.encoded_feature_names))
        ]
        encoded_data = df[encoded_columns]

        original_data = self.encoder.inverse_transform(encoded_data)
        original_df = pd.DataFrame(
            original_data,
            index=df.index,
            columns=self.categorical_features,
            dtype=pd.CategoricalDtype,
        )

        container.__setattr__(
            self.df_key,
            pd.concat([df.drop(encoded_columns, axis=1), original_df], axis=1),
        )
        return container
