from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch as th

from ...utils.logging import logger
from ..containers import BaseContainer
from ..iterators import BaseIterator
from ..utils.utils import DatasetType, train_test_val_split_on_column
from .data_processor import DataProcessor


class DataHandler:
    """
    DataHandler class to handle data loading, splitting, feature extraction and
    dataset iterator creation.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data.
    params : DataHandler.Params
        DataHandler.Params object containing the parameters of the DataHandler.
    """

    @dataclass
    class Params:
        index_column: str
        """Column that should be used as index (unique identifier)"""

        data_processor: DataProcessor
        """DataProcessor object to extract features from the data and create the
            iterator."""

        stratify_on: Optional[str] = None
        """Column on which to stratify the split, by default None.
        If None, no stratification is performed."""

        val_frac: float = 0.2
        """Column on which to stratify the split
            (keeping proportions for each split)
            If None, no stratification is performed"""

        test_frac: float = 0.1
        """Fraction allotted to the validation set, by default 0.2"""

        dryrun: bool = False
        """Dry run mode for fast computation on a large dataset (sampling of a
            small portion), by default False"""

        random_state: Optional[int] = None
        """Random state for reproducibility, by default None"""

        tensors_dtype: Optional[th.dtype] = None
        """Data type of the tensors, by default None"""

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        params: Params,
    ) -> "DataHandler":
        """Initialize DataHandler from csv.

        Parameters
        ----------
        csv_path : str
            Path to the data file.
        params : Params
        Returns
        -------
        DataHandler
            Initialized DataHandler object.
        """
        return cls(
            pd.read_csv(csv_path),
            params,
        )

    def __init__(
        self,
        data: pd.DataFrame,
        params: Params,
    ) -> None:
        self.dryrun = params.dryrun
        self.index_column = params.index_column
        self.data_processor = params.data_processor
        self.stratify_on = params.stratify_on
        self.val_frac = params.val_frac
        self.test_frac = params.test_frac
        self.random_state = params.random_state
        self.tensors_dtype = params.tensors_dtype
        self.full_df = data
        if self.dryrun:
            self.full_df = self.full_df.sample(
                frac=0.05, random_state=self.random_state
            )
        self.full_df.set_index(self.index_column, inplace=True, drop=False)
        self.splits: List[DatasetType] = ["train", "val", "test"]
        self.index_splits: Dict[DatasetType, List[str]] = {}
        self.features: Dict[DatasetType, Dict[str, BaseContainer]] = defaultdict(dict)

    def split_data(
        self,
    ) -> "DataHandler":
        """Split the data into train, test and validation sets,
        split indices are stored in self.index_splits.

        Returns
        -------
        DataHandler
            set
        """

        df_splits = train_test_val_split_on_column(
            self.full_df,
            self.stratify_on,
            val_frac=self.val_frac,
            test_frac=self.test_frac,
            random_state=self.random_state,
        )
        self.index_splits = {
            split: df.index.to_list() for split, df in df_splits.items()
        }
        return self

    def _get_split_iterator(self, split: DatasetType) -> BaseIterator:
        keys = self.index_splits[split]
        iterator = self.data_processor.make_iterator(
            keys=self.index_splits[split], data=self.full_df.loc[keys], split=split
        )
        return iterator

    def get_iterators(self) -> Dict[DatasetType, BaseIterator]:
        """Return a dictionary of iterators for the different splits of the data.

        Returns
        -------
        Dict[DatasetType, BaseDatasetIterator]
            Dictionary of iterators for the different splits of the data.
        """
        if not self.index_splits:
            logger.warning("No data split yet. Splitting data now.")
            self.split_data()
        return {split: self._get_split_iterator(split) for split in self.index_splits}
