from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch as th

from ...extractors import QueryStructureExtractor
from ...utils.query_plan import QueryPlanStructure
from ..conftest import QueryPlanElements


@pytest.fixture
def df_fixture(
    sample_plan_1: QueryPlanElements, sample_plan_2: QueryPlanElements
) -> pd.DataFrame:
    input_df = pd.DataFrame.from_dict(
        {
            "id": [1, 2, 3],
            "tid": [1, 2, 1],
            "plan": [
                sample_plan_1.query_plan,
                sample_plan_2.query_plan,
                sample_plan_1.query_plan,
            ],
        }
    )
    return input_df


class TestStructureExtractor:
    def test_structure_extractor_has_feature_types(self) -> None:
        extractor = QueryStructureExtractor()
        assert extractor.feature_types == {"rows_count": float, "size": float}

    def test_structures_match_templates(self, df_fixture: pd.DataFrame) -> None:
        extractor = QueryStructureExtractor()
        for row in df_fixture.itertuples():
            s_dict = extractor._extract_structure_and_features(
                row.id, row.plan, "train"
            )
            assert set(s_dict.keys()) == {
                "operation_id",
                "operation_gid",
                *extractor.feature_types.keys(),
                *[f"meta_{feature}" for feature in extractor.feature_types.keys()],
            }
            for key, value in s_dict.items():
                if "meta" in key:
                    assert isinstance(value, float)
                else:
                    assert len(value) == len(row.plan.splitlines())
        for plan in extractor.template_plans.values():
            assert type(plan) == QueryPlanStructure
        assert len(extractor.template_plans) == 2
        assert extractor.id_template_dict == {1: 1, 2: 2, 3: 1}

    def test_extract_structure_from_df_returns_correct_shape(
        self, df_fixture: pd.DataFrame
    ) -> None:
        """Graph features and meta features have the correct shape"""
        extractor = QueryStructureExtractor()
        structure_container = extractor.extract_features(df_fixture, "train")

        multi_index = pd.MultiIndex.from_tuples(
            [
                [row.id, i]
                for row in df_fixture.itertuples()
                for i, _ in enumerate(row.plan.splitlines())
            ],
            names=["plan_id", "operation_id"],
        )
        assert (
            graph_meta_features := structure_container.graph_meta_features
        ) is not None
        assert graph_meta_features.shape == (len(df_fixture), 2)

        np.testing.assert_array_equal(
            graph_meta_features.columns, ["rows_count", "size"]
        )
        assert (multi_index == structure_container.graph_features.index).all()

    def test_extract_structure_from_unseen_structure(
        self, df_fixture: pd.DataFrame
    ) -> None:
        """Extracting features from a query plan that is not in the training set"""
        extractor = QueryStructureExtractor()
        structure_container = extractor.extract_features(df_fixture, "val")
        assert len(structure_container.template_plans) == 2
        assert structure_container.graph_meta_features is not None
        assert structure_container.graph_meta_features.shape == (len(df_fixture), 2)

    def test_extract_structure_from_df_returns_correct_values(
        self, df_fixture: pd.DataFrame
    ) -> None:
        """Values in the graph_features and graph_meta_features dataframes
        match values in the dictionary"""
        extractor = QueryStructureExtractor()
        structure_container = extractor.extract_features(df_fixture, "train")
        for row in df_fixture.itertuples():
            features_dict = extractor._extract_structure_and_features(
                row.id, row.plan, "val"
            )
            for feature in ["rows_count", "size"]:
                assert (
                    graph_meta_features := structure_container.graph_meta_features
                ) is not None
                np.testing.assert_allclose(
                    structure_container.graph_features.loc[row.id][feature].values,
                    features_dict[feature],
                    rtol=1e-6,
                )
                assert structure_container.graph_meta_features is not None
                np.testing.assert_allclose(
                    graph_meta_features.loc[row.id][feature],
                    features_dict[f"meta_{feature}"],
                    rtol=1e-6,
                )

    def test_extract_operation_types(self, df_fixture: pd.DataFrame) -> None:
        extractor = QueryStructureExtractor()
        row = df_fixture.iloc[1]
        # avoid error message for template id.
        extractor.extract_features(df_fixture, "train")
        extractor.operation_types = {}
        features_dict = extractor._extract_structure_and_features(
            row.id, row.plan, "train"
        )
        assert features_dict["operation_gid"] == [0, 1, 2, 3, 4]

        row = df_fixture.iloc[0]
        features_dict = extractor._extract_structure_and_features(
            row.id, row.plan, "val"
        )
        assert features_dict["operation_gid"] == [
            -1,
            -1,
            0,
            1,
            2,
            -1,
            2,
            -1,
            2,
            -1,
            2,
            3,
            4,
            2,
            3,
            4,
            2,
            3,
            4,
            2,
            3,
            4,
        ]
        features_dict = extractor._extract_structure_and_features(
            row.id, row.plan, "train"
        )

        assert features_dict["operation_gid"] == [
            5,
            6,
            0,
            1,
            2,
            7,
            2,
            7,
            2,
            7,
            2,
            3,
            4,
            2,
            3,
            4,
            2,
            3,
            4,
            2,
            3,
            4,
        ]

    @pytest.mark.parametrize("positional_encoding_size", [2, 3])
    def test_extract_structure_with_pe(
        self, df_fixture: pd.DataFrame, positional_encoding_size: int
    ) -> None:
        extractor = QueryStructureExtractor(
            positional_encoding_size=positional_encoding_size
        )
        row = df_fixture.iloc[1]
        extractor._extract_structure_and_features(row.id, row.plan, "train")
        result = cast(th.Tensor, extractor.template_plans[1].graph.ndata["pos_enc"])
        th.allclose(
            result,
            th.tensor(
                [
                    [-0.4508, 1.0000, 0.4508],
                    [-0.6977, 0.0000, -0.6977],
                    [-0.5091, 0.0000, 0.5091],
                    [0.0563, 0.0000, 0.0563],
                    [0.2181, 0.0000, -0.2181],
                ]
            )[:, :positional_encoding_size],
        )

    def test_extract_structure_with_pe_at_0(self, df_fixture: pd.DataFrame) -> None:
        extractor = QueryStructureExtractor(positional_encoding_size=0)
        row = df_fixture.iloc[1]
        extractor._extract_structure_and_features(row.id, row.plan, "train")
        assert "pos_enc" not in extractor.template_plans[1].graph.ndata
