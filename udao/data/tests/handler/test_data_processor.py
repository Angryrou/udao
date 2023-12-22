from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from ....utils.interfaces import VarTypes
from ...containers.tabular_container import TabularContainer
from ...extractors.tabular_extractor import TabularFeatureExtractor
from ...handler.data_processor import (
    DataProcessor,
    FeaturePipeline,
    create_data_processor,
)
from ...iterators.tabular_iterator import TabularIterator
from ...preprocessors.normalize_preprocessor import NormalizePreprocessor


@pytest.fixture
def sample_df() -> pd.DataFrame:
    n = 1000
    ids = list(range(1, n + 1))
    tids = [1 for _ in range(n - 10)] + [2 for _ in range(10)]
    values = list(range(n))

    df = pd.DataFrame.from_dict({"id": ids, "tid": tids, "value": values})
    df.set_index("id", inplace=True)

    return df


@pytest.fixture
def data_processor() -> DataProcessor:
    data_processor = DataProcessor(
        iterator_cls=TabularIterator,
        feature_extractors={
            "tabular_feature": TabularFeatureExtractor(
                columns={"value": VarTypes.FLOAT}
            )
        },
        feature_preprocessors={
            "tabular_feature": [NormalizePreprocessor(MinMaxScaler())]
        },
    )

    return data_processor


def test_create_data_processor() -> None:
    # Create the dynamic getter
    data_processor_getter = create_data_processor(TabularIterator)

    scaler = MinMaxScaler()
    # Instantiate the dynamic class
    params_instance = data_processor_getter(
        tabular_feature=FeaturePipeline(
            extractor=TabularFeatureExtractor(columns=["col1", "col2"]),
            preprocessors=[NormalizePreprocessor(scaler)],
        ),
    )

    # Test if the provided parameters exist and are set correctly
    assert params_instance.iterator_cls == TabularIterator
    assert isinstance(
        params_instance.feature_extractors["tabular_feature"], TabularFeatureExtractor
    )

    assert params_instance.feature_processors is not None
    assert len(params_instance.feature_processors) == 1
    assert isinstance(
        params_instance.feature_processors["tabular_feature"][0], NormalizePreprocessor
    )


class TestDataProcessor:
    def test_extract_features_applies_normalization(
        self, sample_df: pd.DataFrame, data_processor: DataProcessor
    ) -> None:
        features = data_processor.extract_features(sample_df, "train")

        assert set(features.keys()) == {"tabular_feature"}
        assert cast(TabularContainer, features["tabular_feature"]).data.shape == (
            len(sample_df),
            1,
        )
        np.testing.assert_array_almost_equal(
            cast(TabularContainer, features["tabular_feature"]).data.values,
            np.linspace(0, 1, len(sample_df)).reshape(-1, 1),
        )

    def test_make_iterators(
        self, sample_df: pd.DataFrame, data_processor: DataProcessor
    ) -> None:
        iterator = data_processor.make_iterator(
            keys=list(sample_df.index), data=sample_df, split="train"
        )
        assert iterator[0] == 0

    def test_inverse_transform(
        self, sample_df: pd.DataFrame, data_processor: DataProcessor
    ) -> None:
        features = data_processor.extract_features(sample_df, "train")
        assert isinstance(features["tabular_feature"], TabularContainer)
        df_inverse = data_processor.inverse_transform(
            container=features["tabular_feature"], pipeline_name="tabular_feature"
        )
        np.testing.assert_array_almost_equal(
            df_inverse[["value"]].values, sample_df[["value"]].values
        )
