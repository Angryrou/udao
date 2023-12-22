from typing import Dict, Optional

import pandas as pd
import pytest

from ....utils.interfaces import VarTypes
from ...containers import TabularContainer
from ...extractors import TabularFeatureExtractor


class TestTabularFeatureExtractor:
    def test_extract_features_with_list(self) -> None:
        # Create a sample DataFrame
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "bool_col": [True, False, True],
                "enum_col": ["a", "b", "a"],
            }
        )
        columns = ["int_col", "float_col"]
        extractor = TabularFeatureExtractor(columns=columns)
        result_container = extractor.extract_features(df)

        # Check if the result is a TabularContainer
        assert isinstance(result_container, TabularContainer)
        # Check if the extracted features contain only the specified columns
        assert list(result_container.data.columns) == columns

    def test_extract_features_with_dict(self) -> None:
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "bool_col": [True, False, True],
                "enum_col": ["a", "b", "a"],
            }
        )
        columns: Dict[str, Optional[VarTypes]] = {
            "int_col": VarTypes.INT,
            "float_col": VarTypes.FLOAT,
            "bool_col": VarTypes.BOOL,
            "enum_col": VarTypes.CATEGORY,
        }
        extractor = TabularFeatureExtractor(columns=columns)
        result_container = extractor.extract_features(df)

        # Check if the result is a TabularContainer
        assert isinstance(result_container, TabularContainer)
        # Check if the extracted DataFrame has the right types
        assert result_container.data["int_col"].dtype == "int"
        assert result_container.data["float_col"].dtype == "float"
        assert result_container.data["bool_col"].dtype == "bool"
        assert result_container.data["enum_col"].dtype.name == "category"

    def test_extract_features_with_invalid_var_type(self) -> None:
        df = pd.DataFrame({"int_col": [1, 2, 3]})
        columns: Dict[str, Optional[VarTypes]] = {"int_col": "INVALID_TYPE"}  # type: ignore
        extractor = TabularFeatureExtractor(columns=columns)

        with pytest.raises(Exception) as exc_info:
            extractor.extract_features(df)
        assert "Unknown variable type" in str(exc_info.value)

    def test_extract_features_with_none_var_type(self) -> None:
        df = pd.DataFrame({"int_col": [1, 2, 3], "float_col": [1.1, 2.2, 3.3]})
        columns = {"int_col": None, "float_col": VarTypes.FLOAT}
        extractor = TabularFeatureExtractor(columns=columns)
        result_container = extractor.extract_features(df)

        assert isinstance(result_container, TabularContainer)
        assert "int_col" in result_container.data.columns
        assert result_container.data["float_col"].dtype == "float"
