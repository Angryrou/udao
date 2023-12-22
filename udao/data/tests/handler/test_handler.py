import random
import string
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch as th

from ...extractors import TabularFeatureExtractor
from ...handler.data_handler import DataHandler
from ...handler.data_processor import DataProcessor
from ...iterators import TabularIterator


def random_string(length: int) -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


@pytest.fixture
def df_fixture() -> Tuple[pd.DataFrame, DataHandler.Params]:
    n = 1000
    ids = list(range(1, n + 1))
    tids = [1 for _ in range(n - 10)] + [2 for _ in range(10)]
    length = [5 for i in range(n)]

    df = pd.DataFrame.from_dict({"id": ids, "tid": tids, "feature": length})

    data_processor = DataProcessor(
        iterator_cls=TabularIterator,
        feature_extractors={"tabular_feature": TabularFeatureExtractor(["feature"])},
    )
    params = DataHandler.Params(
        index_column="id",
        data_processor=data_processor,
        stratify_on=None,
        test_frac=0.1,
        val_frac=0.3,
        random_state=1,
    )
    return df, params


class TestDataHandler:
    def test_split_applies_stratification(
        self, df_fixture: Tuple[pd.DataFrame, DataHandler.Params]
    ) -> None:
        df, params = df_fixture
        params.stratify_on = "tid"
        dh = DataHandler(df, params)
        dh.split_data()
        for split, keys in dh.index_splits.items():
            df_split = df.loc[keys]
            assert len(df_split[df_split["tid"] == 1]) == 99 * len(
                df_split[df_split["tid"] == 2]
            )

    def test_split_no_stratification(
        self, df_fixture: Tuple[pd.DataFrame, DataHandler.Params]
    ) -> None:
        """Check that the split is done correctly
        when no stratification is applied.
        The proportions are correct and there is
        no intersection between the splits."""
        df, params = df_fixture
        params.stratify_on = None
        params.random_state = 1
        dh = DataHandler(df, params)
        dh.split_data()
        assert len(dh.index_splits["test"]) / len(dh.full_df) == dh.test_frac
        assert len(dh.index_splits["val"]) / len(dh.full_df) == dh.val_frac
        assert len(dh.index_splits["train"]) / len(dh.full_df) == 1 - (
            dh.val_frac + dh.test_frac
        )
        assert not set(dh.index_splits["test"]) & set(dh.index_splits["val"])
        assert not set(dh.index_splits["test"]) & set(dh.index_splits["train"])
        assert not set(dh.index_splits["val"]) & set(dh.index_splits["train"])

        df_split = df.loc[dh.index_splits["test"]]
        # results are deterministic because of random_state
        # another random_state would give different results
        assert len(df_split[df_split["tid"] == 1]) == 98
        assert len(df_split[df_split["tid"] == 2]) == 2

    def test_get_iterators_calls_split_data(
        self, df_fixture: Tuple[pd.DataFrame, DataHandler.Params]
    ) -> None:
        df, params = df_fixture
        dh = DataHandler(df, params)
        with patch.object(dh, "split_data") as mock_split_data:
            dh.get_iterators()
            mock_split_data.assert_called_once()

    def test_get_iterators(
        self, df_fixture: Tuple[pd.DataFrame, DataHandler.Params]
    ) -> None:
        np.random.seed(1)
        df, params = df_fixture
        dh = DataHandler(df, params)
        iterators = dh.split_data().get_iterators()
        assert len(iterators) == 3
        assert all(
            isinstance(it, params.data_processor.iterator_cls)
            for it in iterators.values()
        )
        for split, it in iterators.items():
            assert len(it) == len(dh.index_splits[split])
            assert th.eq(it[0], th.tensor(5, dtype=th.float32))
