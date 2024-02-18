from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import dgl
import pandas as pd

from ...data.containers.base_container import BaseContainer
from ..utils.query_plan import QueryPlanStructure


@dataclass
class QueryDescription:
    template_id: int
    template_graph: dgl.DGLGraph
    graph_features: Iterable
    meta_features: Optional[Iterable]
    operation_types: Iterable


@dataclass
class QueryStructureContainer(BaseContainer):
    """Container for the query structure and features of a query plan."""

    graph_features: pd.DataFrame
    """ Stores the features of the operations in the query plan."""
    graph_meta_features: Optional[pd.DataFrame]
    """ Stores the meta features of the operations in the query plan."""
    template_plans: Dict[int, QueryPlanStructure]
    """Link a template id to a QueryPlanStructure"""
    key_to_template: Dict[str, int]
    """Link a key to a template id."""
    operation_types: pd.Series
    """Stores the operation types of the operations in the query plan."""

    def get(self, key: str) -> QueryDescription:
        graph_features = self.graph_features.loc[key].values
        graph_meta_features = (
            None
            if self.graph_meta_features is None
            else self.graph_meta_features.loc[key].values
        )
        template_id = self.key_to_template[key]
        template_graph = self.template_plans[template_id].graph.clone()
        operation_types = self.operation_types.loc[key].values
        return QueryDescription(
            template_id,
            template_graph,
            graph_features,
            graph_meta_features,
            operation_types,
        )
