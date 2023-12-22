from typing import Dict, List, Optional, Union

import pandas as pd

from ...utils.interfaces import VarTypes
from ..containers import TabularContainer
from .base_extractors import StaticExtractor


class TabularFeatureExtractor(StaticExtractor[TabularContainer]):
    """
    Extract columns from a DataFrame as a TabularContainer.

    Parameters
    ----------
    columns : Union[List[str], Dict[str, Optional[VarTypes]]], optional
        Either:
        - a list of column names to extract from the DataFrame
        - a dictionary that maps column names to variable types
        if the variable type is None, the column is extracted
        without casting
        - None, in which case all columns are extracted

    """

    def __init__(
        self, columns: Optional[Union[List[str], Dict[str, Optional[VarTypes]]]] = None
    ) -> None:
        self.columns = columns

    def extract_features(self, df: pd.DataFrame) -> TabularContainer:
        """extract and cast features from the DataFrame"""
        if isinstance(self.columns, list):
            extracted_df = df[self.columns]
        elif isinstance(self.columns, dict):
            extracted_df = pd.DataFrame()
            for col, var_type in self.columns.items():
                if isinstance(var_type, VarTypes):
                    extracted_df[col] = df[col].astype(var_type.value)
                elif var_type is None:
                    extracted_df[col] = df[col]
                else:
                    raise Exception(f"Unknown variable type: {var_type}")
        else:
            extracted_df = df

        return TabularContainer(extracted_df)
