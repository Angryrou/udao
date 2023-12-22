import pandas as pd
import pytest

from ...predicate_embedders.utils import extract_operations, prepare_operation


@pytest.mark.parametrize(
    "op,expected",
    [
        (
            "GlobalLimit 10, Statistics(sizeInBytes=400.0 B, rowCount=10)",
            "globallimit 10",
        ),
        (
            "+- LocalLimit 10, Statistics(sizeInBytes=6.9 GiB, rowCount=1.86E+8)",
            "locallimit 10",
        ),
        (
            "   +- Sort [revenue#47608 DESC NULLS LAST, "
            "o_orderdate#22 ASC NULLS FIRST], true, Statistics(sizeInBytes=6.9 GiB,"
            " rowCount=1.86E+8)",
            "sort revenue desc nulls last o_orderdate asc nulls first true",
        ),
        (
            "      +- Aggregate [l_orderkey#23L, o_orderdate#22,"
            " o_shippriority#20], [l_orderkey#23L, sum(CheckOverflow"
            "((promote_precision(cast(l_extendedprice#28 as "
            "decimal(13,2))) * promote_precision(CheckOverflow((1.00"
            " - promote_precision(cast(l_discount#29 as decimal(13,2))))"
            ", DecimalType(13,2), true))), DecimalType(26,4), true)) AS"
            " revenue#47608, o_orderdate#22, o_shippriority#20],"
            " Statistics(sizeInBytes=6.9 GiB, rowCount=1.86E+8)",
            "aggregate l_orderkey o_orderdate o_shippriority l_orderkey"
            " sum checkoverflow promote_precision cast l_extendedprice as"
            " decimal 13 2 rmul promote_precision checkoverflow 1.00 rminus"
            " promote_precision cast l_discount as decimal 13 2 decimaltype 13 2"
            " true decimaltype 26 4 true as revenue o_orderdate o_shippriority",
        ),
        (
            "               +- Relation tpch_100.part[p_partkey#59113L,p_name#59114"
            ",p_mfgr#59115,p_type#59116,p_size#59117,p_container#59118,p_retailprice"
            "#59119,p_comment#59120,p_brand#59121] parquet, Statistics(sizeInBytes=3.5"
            " GiB, rowCount=1.92E+7)",
            "relation tpch_100.part p_partkey p_name p_mfgr p_type p_size p_container"
            " p_retailprice p_comment p_brand parquet",
        ),
    ],
)
def test_prepare_operation(op: str, expected: str) -> None:
    prepared = prepare_operation(op)
    assert prepared == expected


@pytest.fixture
def df_fixture() -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        {
            "id": [1, 2],
            "plan": [
                "a b c\na b d",
                "a b d\na b x",
            ],
        }
    )


def test_extract_operations(df_fixture: pd.DataFrame) -> None:
    plan_to_op, operations = extract_operations(df_fixture)
    assert plan_to_op == {1: [0, 1], 2: [1, 2]}
    assert operations == ["a b c", "a b d", "a b x"]


def test_extract_operations_processing_is_applied(df_fixture: pd.DataFrame) -> None:
    plan_to_op, operations = extract_operations(
        df_fixture, operation_processing=lambda s: s.replace("x", "c")
    )
    assert plan_to_op == {1: [0, 1], 2: [1, 0]}
    assert operations == ["a b c", "a b d"]
