from typing import Dict, List, Optional, Tuple

import pandas as pd

from ...utils.logging import logger
from ..containers import QueryStructureContainer
from ..utils.query_plan import (
    QueryPlanOperationFeatures,
    QueryPlanStructure,
    add_positional_encoding,
    extract_query_plan_features,
)
from ..utils.utils import DatasetType
from .base_extractors import TrainedExtractor


class QueryStructureExtractor(TrainedExtractor[QueryStructureContainer]):
    """
    Extracts the features of the operations in the logical plan,
    and the tree structure of the logical plan.
    Keep track of the different query plans seen so far, and their template id.

    Parameters
    ----------
    with_positional_encoding: bool
        Whether to add positional encoding to the query plan gaph.
    """

    def __init__(self, positional_encoding_size: Optional[int] = None) -> None:
        self.template_plans: Dict[int, QueryPlanStructure] = {}
        self.feature_types: Dict[
            str, type
        ] = QueryPlanOperationFeatures.get_feature_names_and_types()
        self.id_template_dict: Dict[str, int] = {}
        self.operation_types: Dict[str, int] = {}
        self.positional_encoding_size = positional_encoding_size

    def _extract_operation_types(
        self, structure: QueryPlanStructure, split: DatasetType
    ) -> List[int]:
        """Find ids of operation types in the query plan.
        Add new operation types if train."""
        operation_gids: List[int] = []
        for name in structure.node_id2name.values():
            op_type = name.split()[0]
            if op_type not in self.operation_types and split == "train":
                self.operation_types[op_type] = len(self.operation_types)
            operation_gids.append(self.operation_types.get(op_type, -1))
        return operation_gids

    def _extract_structure_template(
        self, structure: QueryPlanStructure, split: DatasetType
    ) -> int:
        """Find template id of the query plan, or create a new one (if train)"""
        tid = None
        for template_id, template_structure in self.template_plans.items():
            if structure.graph_match(template_structure):
                tid = template_id
                break

        if tid is None:
            tid = len(self.template_plans) + 1
            if self.positional_encoding_size:
                structure.graph = add_positional_encoding(
                    structure.graph, self.positional_encoding_size
                )
            self.template_plans[tid] = structure
            if split != "train":
                logger.warning(
                    f"Unknown template plan in test or validation set, set as {tid}"
                )
        return tid

    def _extract_structure_and_features(
        self, idx: str, query_plan: str, split: DatasetType
    ) -> Dict:
        """Extract the features of the operations in the logical plan,
        and the tree structure of the logical plan.

        Parameters
        ----------
        idx: str
            The id of the query plan.
        query_plan : str
            A query plan string.

        Returns
        -------
        Dict
            - template_id: id of the template of the query plan
            - operation_id: list of operation ids in the query plan
            - one key per feature for features of the operations
            in the query plan

        """
        structure, op_features, meta_features = extract_query_plan_features(query_plan)
        operation_gids = self._extract_operation_types(structure, split)
        self.id_template_dict[idx] = self._extract_structure_template(structure, split)
        return {
            "operation_id": op_features.operation_ids,
            "operation_gid": operation_gids,
            **op_features.features_dict,
            **meta_features,
        }

    def _derive_meta_dataframe(
        self, df_op_features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create meta dataframe from the meta_ columns
        of the df_op_features dataframe. These features are aggregate
        of operations and thusgive one line per query plan."""
        df_meta_features = df_op_features[
            ["plan_id"]
            + [col for col in df_op_features.columns if col.startswith("meta_")]
        ]
        df_meta_features = df_meta_features.rename(
            columns=lambda x: x.replace("meta_", "")
        )
        df_meta_features.set_index("plan_id", inplace=True)
        filtered_df_op_features = df_op_features[
            [col for col in df_op_features.columns if not col.startswith("meta_")]
        ]
        return filtered_df_op_features, df_meta_features

    def _extract_op_features_exploded(
        self, df_op_features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Explode the df_op_features dataframe to have one row per operation
        in the query plans, and one column per feature of the operations."
        """
        df_op_features_exploded = df_op_features.explode(
            ["operation_id", "operation_gid"] + list(self.feature_types.keys()),
            ignore_index=True,
        )
        df_op_features_exploded = df_op_features_exploded.set_index(
            ["plan_id", "operation_id"]
        )
        df_op_features_exploded[
            list(self.feature_types.keys())
        ] = df_op_features_exploded[list(self.feature_types.keys())].astype("float32")
        df_operation_types = df_op_features_exploded["operation_gid"].astype("int32")
        del df_op_features_exploded["operation_gid"]
        return df_op_features_exploded, df_operation_types

    def extract_features(
        self, df: pd.DataFrame, split: DatasetType
    ) -> QueryStructureContainer:
        """Extract the features of the operations in the logical plan,
        and the tree structure of the logical plan for each query plan
        in the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with a column "plan" containing the query plans.

        Returns
        -------
        pd.DataFrame
            Dataframe with one row per operation in the query plans,
            and one column per feature of the operations.
        """
        df_op_features: pd.DataFrame = df.apply(
            lambda row: self._extract_structure_and_features(row.id, row.plan, split),
            axis=1,
        ).apply(pd.Series)
        df_op_features["plan_id"] = df["id"]
        df_op_features, df_meta_features = self._derive_meta_dataframe(df_op_features)
        (
            df_op_features_exploded,
            df_operation_types,
        ) = self._extract_op_features_exploded(df_op_features)
        return QueryStructureContainer(
            graph_features=df_op_features_exploded,
            template_plans=self.template_plans,
            key_to_template=self.id_template_dict,
            graph_meta_features=df_meta_features,
            operation_types=df_operation_types,
        )
