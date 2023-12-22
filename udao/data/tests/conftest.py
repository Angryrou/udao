import json
from pathlib import Path
from typing import Dict, NamedTuple

import pytest

from ..utils.query_plan import (
    QueryPlanOperationFeatures,
    QueryPlanStructure,
    compute_meta_features,
    format_size,
)

QueryPlanElements = NamedTuple(
    "QueryPlanElements",
    [
        ("query_plan", str),
        ("structure", QueryPlanStructure),
        ("features", QueryPlanOperationFeatures),
        ("meta_features", Dict[str, float]),
    ],
)


def get_query_plan_sample(
    json_path: str,
) -> QueryPlanElements:
    base_dir = Path(__file__).parent
    with open(base_dir / json_path, "r") as f:
        plan_features = json.load(f)
    incoming_ids = plan_features["incoming_ids"]
    outgoing_ids = plan_features["outgoing_ids"]
    names = plan_features["names"]
    row_counts = [float(v) for v in plan_features["row_counts"]]
    sizes = [format_size(s) for s in plan_features["sizes"]]
    structure = QueryPlanStructure(names, incoming_ids, outgoing_ids)
    operation_features = QueryPlanOperationFeatures(row_counts, sizes)
    meta_features = compute_meta_features(structure, operation_features)
    return QueryPlanElements(
        plan_features["query_plan"], structure, operation_features, meta_features
    )


@pytest.fixture(scope="session")
def sample_plan_1() -> QueryPlanElements:
    base_dir = Path(__file__).parent
    return get_query_plan_sample(str(base_dir / "assets/sample_plan_1.json"))


@pytest.fixture(scope="session")
def sample_plan_2() -> QueryPlanElements:
    base_dir = Path(__file__).parent
    return get_query_plan_sample(str(base_dir / "assets/sample_plan_2.json"))
