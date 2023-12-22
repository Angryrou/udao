from typing import List

import numpy as np
import pytest

from ...utils.moo_utils import summarize_ret


@pytest.mark.parametrize(
    "po_obj_list, po_var_list, expected_obj, expected_var",
    [
        (
            [[1, 1], [1, 2], [2, 1]],
            [[0.5, 0.3], [0.9, 2.5], [2.1, 1.5]],
            [[1, 1]],
            [[0.5, 0.3]],
        ),
        (
            [[2, 1], [1, 1], [1, 4], [3, 2], [0.5, 2.1], [0.4, 1.9]],
            [[0.5, 0.3], [0.9, 2.5], [2.1, 1.5], [1.1, 1.2], [0.5, 0.3], [0.3, 21.5]],
            [[1, 1], [0.4, 1.9]],
            [[0.9, 2.5], [0.3, 21.5]],
        ),
        (
            [[2, 6], [3, 5]],
            [[0.5, 0.3], [0.9, 2.5]],
            [[2, 6], [3, 5]],
            [[0.5, 0.3], [0.9, 2.5]],
        ),
    ],
)
def test_summarize_ret(
    po_obj_list: List[List[float]],
    po_var_list: List[List[float]],
    expected_obj: List[List[float]],
    expected_var: List[List[float]],
) -> None:
    po_objs, po_vars = summarize_ret(po_obj_list, po_var_list)
    np.testing.assert_array_equal(po_objs, np.array(expected_obj))
    np.testing.assert_array_equal(po_vars, expected_var)
