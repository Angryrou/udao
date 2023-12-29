from typing import Dict, Literal, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

PandasTypes = {float: "float64", int: "int64", str: "object"}

DatasetType = Literal["train", "val", "test"]


def train_test_val_split_on_column(
    df: pd.DataFrame,
    groupby_col: Optional[str],
    *,
    val_frac: float,
    test_frac: float,
    random_state: Optional[int] = None,
) -> Dict[DatasetType, pd.DataFrame]:
    """return a dictionary of DatasetType (train/val/test) and the DataFrame"""
    train_df, non_train_df = train_test_split(
        df,
        test_size=val_frac + test_frac,
        stratify=df[groupby_col] if groupby_col else None,
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        non_train_df,
        test_size=test_frac / (val_frac + test_frac),
        stratify=non_train_df[groupby_col] if groupby_col else None,
        random_state=random_state,
    )
    df_dict: Dict[DatasetType, pd.DataFrame] = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    return df_dict
