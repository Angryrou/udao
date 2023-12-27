import re
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import pandas as pd


def remove_unknown(s: str) -> str:
    """Remove unknown symbol from a query plan
    (in the form of (unknown))
    """
    pattern = r"\(unknown\)"
    # Remove unknown operations
    s = re.sub(pattern, "", s)
    return s


def remove_statistics(s: str) -> str:
    """Remove statistical information from a query plan
    (in the form of Statistics(...)
    """
    pattern = r"\bStatistics\([^)]+\)"
    # Remove statistical information
    s = re.sub(pattern, "", s)
    return s


def remove_hashes(s: str) -> str:
    """Remove hashes from a query plan, e.g. #1234L"""
    # Replace hashes with a placeholder or remove them
    return re.sub(r"#[0-9]+[L]*", "", s)


def brief_clean(s: str) -> str:
    """Remove special characters from a string and convert to lower case"""
    return re.sub(r"[^0-9A-Za-z\'_.]+", " ", s).lower()


def replace_symbols(s: str) -> str:
    """Replace symbols with tokens"""
    return (
        s.replace(" >= ", " GE ")
        .replace(" <= ", " LE ")
        .replace(" == ", " EQ")
        .replace(" = ", " EQ ")
        .replace(" > ", " GT ")
        .replace(" < ", " LT ")
        .replace(" != ", " NEQ ")
        .replace(" + ", " rADD ")
        .replace(" - ", " rMINUS ")
        .replace(" / ", " rDIV ")
        .replace(" * ", " rMUL ")
    )


def remove_duplicate_spaces(s: str) -> str:
    return " ".join(s.split())


def prepare_operation(operation: str) -> str:
    """Prepare an operation for embedding by keeping only
    relevant semantic information"""
    processings: List[Callable[[str], str]] = [
        remove_unknown,
        remove_statistics,
        remove_hashes,
        replace_symbols,
        brief_clean,
        remove_duplicate_spaces,
    ]
    for processing in processings:
        operation = processing(operation)
    return operation


def build_unique_operations(df: pd.DataFrame) -> Tuple[Dict[int, List[int]], List[str]]:
    """Build a dictionary of unique operations and their IDs"""
    unique_ops: Dict[str, int] = defaultdict(lambda: len(unique_ops))
    plan_to_ops: Dict[int, List[int]] = defaultdict(list)
    for row in df.itertuples():
        plan_to_ops[row.id].append(unique_ops[row.operation])

    operations_list = list(unique_ops.keys())
    return plan_to_ops, operations_list


def extract_operations(
    plan_df: pd.DataFrame, operation_processing: Callable[[str], str] = lambda x: x
) -> Tuple[Dict[int, List[int]], List[str]]:
    """Extract unique operations from a DataFrame of
    query plans and links them to query plans.
    Operations are transformed using prepare_operation
    to remove statistical information and hashes.

    Parameters
    ----------
    plan_df : pd.DataFrame
        DataFrame containing the query plans and their ids.

    operation_processing : Callable[[str], str]
        Function to process the operations, by default no processing will be applied
        and the raw operations will be used.

    Returns
    -------
    Tuple[Dict[int, List[int]], List[str]]
        plan_to_ops: Dict[int, List[int]]
            Links a query plan ID to a list of operation IDs in the operations list
        operations_list: List[str]
            List of unique operations in the dataset
    """
    df = plan_df[["id", "plan"]].copy()

    df["plan"] = df["plan"].apply(
        lambda plan: [operation_processing(op) for op in plan.splitlines()]  # type: ignore
    )
    df = df.explode("plan", ignore_index=True)
    df.rename(columns={"plan": "operation"}, inplace=True)
    return build_unique_operations(df)
