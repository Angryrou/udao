from typing import Dict, Iterable, Optional

import numpy as np
import pytest
import torch as th

from ... import concepts as co
from ...soo.random_sampler_solver import RandomSamplerSolver


class TestRandomSampler:
    @pytest.mark.parametrize(
        "test_data, expected",
        [
            ({"variable": co.BoolVariable(), "n_samples": 3}, [0, 1, 1]),
            (
                {"variable": co.IntegerVariable(1, 7), "n_samples": 2},
                [5, 6],
            ),
            (
                {"variable": co.FloatVariable(2, 4), "n_samples": 5},
                [3.098, 3.430, 3.206, 3.090, 2.847],
            ),
            (
                {
                    "variable": co.EnumVariable([0, 4, 7, 10]),
                    "n_samples": 2,
                    "range": [0, 4, 7, 10],
                },
                [0, 10],
            ),
        ],
    )
    def test_random_sampler_single_variable(
        self, test_data: Dict, expected: Iterable
    ) -> None:
        solver = RandomSamplerSolver(
            RandomSamplerSolver.Params(n_samples_per_param=test_data["n_samples"])
        )
        output = solver._get_input(variables={"v1": test_data["variable"]}, seed=0)
        np.testing.assert_allclose(
            [[o] for o in output["v1"]], [[e] for e in expected], rtol=1e-2
        )

    def test_random_sampler_multiple_variables(self) -> None:
        solver = RandomSamplerSolver(RandomSamplerSolver.Params(n_samples_per_param=3))

        output = solver._get_input(
            variables={"v1": co.BoolVariable(), "v2": co.IntegerVariable(1, 7)}, seed=0
        )
        expected_array = np.array(
            [
                [0, 5],
                [1, 6],
                [1, 1],
            ]
        )
        np.testing.assert_equal(
            output, {"v1": expected_array[:, 0], "v2": expected_array[:, 1]}
        )

    def test_solve(self) -> None:
        solver = RandomSamplerSolver(RandomSamplerSolver.Params(n_samples_per_param=50))

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
        soo_obj, soo_vars = solver.solve(problem, seed=0)
        assert soo_obj == 8
        assert soo_vars == {"v1": 1, "v2": 7}
