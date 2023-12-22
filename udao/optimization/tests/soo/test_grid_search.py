from typing import Dict, Iterable, Optional

import numpy as np
import pytest
import torch as th

from ... import concepts as co
from ...soo.grid_search_solver import GridSearchSolver


class TestGridSearch:
    @pytest.mark.parametrize(
        "test_data, expected",
        [
            ({"variable": co.BoolVariable(), "n_grids": 1}, [0]),
            ({"variable": co.BoolVariable(), "n_grids": 2}, [0, 1]),
            ({"variable": co.BoolVariable(), "n_grids": 3}, [0, 1]),
            (
                {"variable": co.IntegerVariable(1, 7), "n_grids": 5},
                [1, 2, 4, 6, 7],
            ),
            (
                {"variable": co.IntegerVariable(1, 7), "n_grids": 8},
                [1, 2, 3, 4, 5, 6, 7],
            ),
            (
                {"variable": co.FloatVariable(2, 4), "n_grids": 5},
                [2, 2.5, 3, 3.5, 4],
            ),
            (
                {"variable": co.EnumVariable([0, 4, 7, 10]), "n_grids": 2},
                [0, 4, 7, 10],
            ),
        ],
    )
    def test_grid_search_get_single_variable(
        self, test_data: Dict, expected: Iterable
    ) -> None:
        solver = GridSearchSolver(
            GridSearchSolver.Params(n_grids_per_var=[test_data["n_grids"]])
        )
        output = solver._get_input(
            variables={"variable": test_data["variable"]},
        )
        np.testing.assert_equal(output, {"variable": np.array([e for e in expected])})

    def test_grid_search_get_multiple_variables(self) -> None:
        solver = GridSearchSolver(GridSearchSolver.Params(n_grids_per_var=[2, 7]))
        output = solver._get_input(
            variables={"v1": co.BoolVariable(), "v2": co.IntegerVariable(1, 7)},
        )
        expected_array = np.array(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [0, 6],
                [0, 7],
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [1, 6],
                [1, 7],
            ]
        ).T
        np.testing.assert_equal(
            output, {"v1": expected_array[0], "v2": expected_array[1]}
        )

    def test_solve(self) -> None:
        solver = GridSearchSolver(GridSearchSolver.Params(n_grids_per_var=[2, 7]))

        def obj1_func(
            input_variables: Dict[str, th.Tensor],
            input_parameters: Optional[Dict[str, th.Tensor]] = None,
        ) -> th.Tensor:
            return input_variables["v1"] + input_variables["v2"]

        objective = co.Objective("obj1", minimize=False, function=obj1_func)
        variables: Dict[str, co.Variable] = {
            "v1": co.BoolVariable(),
            "v2": co.IntegerVariable(1, 7),
        }
        problem = co.SOProblem(objective=objective, variables=variables, constraints=[])
        soo_obj, soo_vars = solver.solve(problem)
        assert soo_obj == 8
        assert soo_vars == {"v1": 1, "v2": 7}
