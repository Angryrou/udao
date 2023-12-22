import numpy as np
import pandas as pd
import pytest

from ...containers.tabular_container import TabularContainer
from ...preprocessors.one_hot_preprocessor import OneHotPreprocessor


@pytest.fixture
def sample_container() -> TabularContainer:
    return TabularContainer(
        pd.DataFrame(
            {
                "color": ["red", "green", "blue", "green"],
                "size": ["S", "M", "L", "S"],
                "price": [10, 15, 20, 15],
            }
        )
    )


@pytest.fixture
def one_hot_encoder_preprocessor() -> OneHotPreprocessor:
    return OneHotPreprocessor(["color", "size"])


class TestOneHotPreprocessor:
    def test_preprocess(
        self,
        one_hot_encoder_preprocessor: OneHotPreprocessor,
        sample_container: TabularContainer,
    ) -> None:
        # Act
        container_encoded = one_hot_encoder_preprocessor.preprocess(
            sample_container, split="train"
        )
        df_encoded = container_encoded.data

        # Assert
        assert "color" not in df_encoded.columns
        assert "size" not in df_encoded.columns
        assert one_hot_encoder_preprocessor.encoded_feature_names is not None
        assert len(one_hot_encoder_preprocessor.encoded_feature_names) > 0
        for feature in one_hot_encoder_preprocessor.encoded_feature_names:
            assert feature in df_encoded.columns

    def test_inverse_preprocess(
        self,
        one_hot_encoder_preprocessor: OneHotPreprocessor,
        sample_container: TabularContainer,
    ) -> None:
        container_encoded = one_hot_encoder_preprocessor.preprocess(
            sample_container, split="train"
        )

        container_original = one_hot_encoder_preprocessor.inverse_preprocess(
            container_encoded
        )
        df_original = container_original.data

        np.testing.assert_array_equal(
            df_original.sort_index(axis=1).values,
            sample_container.data.sort_index(axis=1).values,
        )

    def test_inverse_preprocess_without_preprocess(
        self,
        one_hot_encoder_preprocessor: OneHotPreprocessor,
        sample_container: TabularContainer,
    ) -> None:
        # Assert
        with pytest.raises(ValueError):
            one_hot_encoder_preprocessor.inverse_preprocess(sample_container)
