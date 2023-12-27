from typing import Dict, Optional

import numpy as np
import pytest
import torch as th

from ....utils.logging import logger
from ...concepts import Constraint, Objective
from ...concepts.problem import MOProblem
from ...moo.weighted_sum import WeightedSum
from ...soo.grid_search_solver import GridSearchSolver
from ...soo.mogd import MOGD
from ...soo.random_sampler_solver import RandomSamplerSolver
from ...soo.so_solver import SOSolver
from ...utils.exceptions import NoSolutionError
from ...utils.moo_utils import even_weights


class TestWeightedSum:
    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearchSolver(GridSearchSolver.Params(n_grids_per_var=[2, 7])),
            RandomSamplerSolver(RandomSamplerSolver.Params(n_samples_per_param=30)),
        ],
    )
    def test_solve_without_input_parameters(
        self, inner_solver: SOSolver, simple_problem: MOProblem
    ) -> None:
        """solve a dummy minimization problem with 2 objectives and 1 constraint"""
        ws_pairs = np.array([[0.3, 0.7], [0.6, 0.4]])
        simple_problem.input_parameters = None

        ws_algo = WeightedSum(
            WeightedSum.Params(
                so_solver=inner_solver,
                ws_pairs=ws_pairs,
            )
        )
        po_objs, po_vars = ws_algo.solve(problem=simple_problem, seed=0)
        np.testing.assert_array_almost_equal(po_objs, np.array([[0, 0.2]]))
        np.testing.assert_equal(po_vars, np.array({"v1": 0, "v2": 2}))

    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearchSolver(GridSearchSolver.Params(n_grids_per_var=[2, 7])),
            RandomSamplerSolver(RandomSamplerSolver.Params(n_samples_per_param=30)),
        ],
    )
    def test_solve_with_input_parameters(
        self, inner_solver: SOSolver, simple_problem: MOProblem
    ) -> None:
        """solve a dummy minimization problem with 2 objectives and 1 constraint"""
        ws_pairs = np.array([[0.3, 0.7], [0.6, 0.4]])

        ws_algo = WeightedSum(
            WeightedSum.Params(
                so_solver=inner_solver,
                ws_pairs=ws_pairs,
            )
        )
        po_objs, po_vars = ws_algo.solve(problem=simple_problem, seed=0)

        np.testing.assert_almost_equal(po_objs, np.array([[1, 1.3]]))
        np.testing.assert_equal(po_vars, np.array([{"v1": 0, "v2": 3}]))

    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearchSolver(GridSearchSolver.Params(n_grids_per_var=[2, 7])),
            RandomSamplerSolver(RandomSamplerSolver.Params(n_samples_per_param=1000)),
        ],
    )
    def test_solver_with_two_obj_problem(
        self, inner_solver: SOSolver, two_obj_problem: MOProblem
    ) -> None:
        ws_pairs = np.array(
            [
                [0.3, 0.7],
                [0.6, 0.4],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.4, 0.6],
                [0.5, 0.5],
            ]
        )

        ws_algo = WeightedSum(
            WeightedSum.Params(
                so_solver=inner_solver,
                ws_pairs=ws_pairs,
            )
        )
        po_objs, po_vars = ws_algo.solve(problem=two_obj_problem, seed=0)

        np.testing.assert_almost_equal(po_objs, np.array([[0, 0]]), decimal=5)
        np.testing.assert_almost_equal(po_vars[0]["v1"], 0.0, decimal=3)
        assert po_vars[0]["v2"] == 1.0

    @pytest.mark.parametrize(
        "strict_rounding",
        [
            True,
            False,
        ],
    )
    def test_solver_with_two_obj_problem_mogd(
        self, strict_rounding: bool, two_obj_problem: MOProblem
    ) -> None:
        inner_solver = MOGD(
            MOGD.Params(
                learning_rate=0.1,
                max_iters=100,
                patience=20,
                multistart=2,
                batch_size=10,
                strict_rounding=strict_rounding,
            )
        )

        ws_pairs = even_weights(0.1, 2)
        logger.debug(f"ws_pairs: {ws_pairs}")
        ws_algo = WeightedSum(
            WeightedSum.Params(
                so_solver=inner_solver,
                ws_pairs=ws_pairs,
                normalize=False,
            )
        )
        po_objs, po_vars = ws_algo.solve(problem=two_obj_problem, seed=0)

        np.testing.assert_almost_equal(po_objs, np.array([[0, 0]]))
        np.testing.assert_almost_equal(po_vars[0]["v1"], 0.0)
        assert po_vars[0]["v2"] == 1.0

    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearchSolver(GridSearchSolver.Params(n_grids_per_var=[2, 7])),
            RandomSamplerSolver(RandomSamplerSolver.Params(n_samples_per_param=1000)),
        ],
    )
    def test_solver_with_two_obj_problem_with_cache(
        self, inner_solver: SOSolver, two_obj_problem: MOProblem
    ) -> None:
        ws_pairs = np.array(
            [
                [0.3, 0.7],
                [0.6, 0.4],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.4, 0.6],
                [0.5, 0.5],
            ]
        )

        ws_algo = WeightedSum(
            WeightedSum.Params(
                so_solver=inner_solver,
                ws_pairs=ws_pairs,
                allow_cache=True,
            )
        )
        po_objs, po_vars = ws_algo.solve(problem=two_obj_problem, seed=0)

        np.testing.assert_almost_equal(po_objs, np.array([[0, 0]]), decimal=3)
        np.testing.assert_almost_equal(po_vars[0]["v1"], 0.0, decimal=2)
        assert po_vars[0]["v2"] == 1.0

    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearchSolver(GridSearchSolver.Params(n_grids_per_var=[2, 7])),
            RandomSamplerSolver(RandomSamplerSolver.Params(n_samples_per_param=10)),
        ],
    )
    def test_ws_raises_no_solution(
        self, inner_solver: SOSolver, simple_problem: MOProblem
    ) -> None:
        ws_pairs = np.array([[0.3, 0.7], [0.6, 0.4]])

        def f3(
            input_variables: Dict[str, th.Tensor],
            input_parameters: Optional[Dict[str, th.Tensor]] = None,
        ) -> th.Tensor:
            return input_variables["v1"] + input_variables["v2"] - 10

        simple_problem.constraints = [Constraint(function=f3, lower=0)]
        simple_problem.input_parameters = None
        ws_algo = WeightedSum(
            WeightedSum.Params(
                so_solver=inner_solver,
                ws_pairs=ws_pairs,
            )
        )
        with pytest.raises(NoSolutionError):
            ws_algo.solve(problem=simple_problem, seed=0)

    @pytest.mark.parametrize(
        "inner_solver",
        [
            GridSearchSolver(GridSearchSolver.Params(n_grids_per_var=[2, 7])),
            RandomSamplerSolver(RandomSamplerSolver.Params(n_samples_per_param=30)),
        ],
    )
    def test_works_with_three_objectives(
        self, inner_solver: SOSolver, simple_problem: MOProblem
    ) -> None:
        ws_pairs = np.array([[0.3, 0.5, 0.2], [0.6, 0.3, 0.1]])

        def f2(
            input_variables: Dict[str, th.Tensor],
            input_parameters: Optional[Dict[str, th.Tensor]] = None,
        ) -> th.Tensor:
            return input_variables["v2"]

        objectives = list(simple_problem.objectives)
        objectives.insert(1, Objective("obj3", function=f2, minimize=True))
        simple_problem.objectives = objectives
        simple_problem.input_parameters = None

        def constraint_f(
            input_variables: Dict[str, th.Tensor],
            input_parameters: Optional[Dict[str, th.Tensor]] = None,
        ) -> th.Tensor:
            return input_variables["v1"] + input_variables["v2"] - 3

        simple_problem.constraints = [Constraint(function=constraint_f, lower=0)]
        ws_algo = WeightedSum(
            WeightedSum.Params(
                so_solver=inner_solver,
                ws_pairs=ws_pairs,
            )
        )
        po_objs, po_vars = ws_algo.solve(problem=simple_problem, seed=0)

        np.testing.assert_almost_equal(po_objs, np.array([[0, 3, 0.3]]))
        np.testing.assert_equal(po_vars, np.array([{"v1": 0, "v2": 3}]))
