from typing import cast

import numpy as np
import pytest
import torch as th

from ....data.handler.data_processor import DataProcessor
from ....model.utils.utils import set_deterministic_torch
from ...concepts.problem import MOProblem
from ...moo.progressive_frontier import ParallelProgressiveFrontier
from ...soo.mogd import MOGD
from ...utils.moo_utils import Point, Rectangle


@pytest.fixture
def ppf(data_processor: DataProcessor, mogd: MOGD) -> ParallelProgressiveFrontier:
    ppf = ParallelProgressiveFrontier(
        params=ParallelProgressiveFrontier.Params(
            processes=1,
            n_grids=2,
            max_iters=4,
        ),
        solver=mogd,
    )
    return ppf


class TestParallelProgressiveFrontier:
    def test_create_grid_cells(self, ppf: ParallelProgressiveFrontier) -> None:
        utopia = Point(np.array([0, 2, 0]))
        nadir = Point(np.array([4, 10, 1]))
        grid_rectangles = ppf._create_grid_cells(utopia, nadir, 2, 3)

        assert len(grid_rectangles) == 8
        expected = [
            Rectangle(
                utopia=Point(objs=np.array([0.0, 2.0, 0.0])),
                nadir=Point(objs=np.array([2.0, 6.0, 0.5])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([0.0, 2.0, 0.5])),
                nadir=Point(objs=np.array([2.0, 6.0, 1.0])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([0.0, 6.0, 0.0])),
                nadir=Point(objs=np.array([2.0, 10.0, 0.5])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([0.0, 6.0, 0.5])),
                nadir=Point(objs=np.array([2.0, 10.0, 1.0])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([2.0, 2.0, 0.0])),
                nadir=Point(objs=np.array([4.0, 6.0, 0.5])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([2.0, 2.0, 0.5])),
                nadir=Point(objs=np.array([4.0, 6.0, 1.0])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([2.0, 6.0, 0.0])),
                nadir=Point(objs=np.array([4.0, 10.0, 0.5])),
            ),
            Rectangle(
                utopia=Point(objs=np.array([2.0, 6.0, 0.5])),
                nadir=Point(objs=np.array([4.0, 10.0, 1.0])),
            ),
        ]
        for i, rect in enumerate(expected):
            assert rect == grid_rectangles[i]

    def test_solve_with_two_objectives(
        self, ppf: ParallelProgressiveFrontier, two_obj_problem: MOProblem
    ) -> None:
        set_deterministic_torch(0)
        objectives, variables = ppf.solve(
            problem=two_obj_problem,
            seed=0,
        )
        assert objectives is not None
        cast(MOGD, ppf.solver).patience = 100
        np.testing.assert_array_equal(objectives, np.array([[0, 0]]))
        assert variables is not None
        assert variables[0] == {"v1": 0.0, "v2": 1.0}

    def test_solve_with_two_objectives_gpu(
        self, ppf: ParallelProgressiveFrontier, two_obj_problem: MOProblem
    ) -> None:
        if not th.cuda.is_available():
            pytest.skip("Skip GPU test")
        cast(MOGD, ppf.solver).device = th.device("cuda")
        set_deterministic_torch(0)
        objectives, variables = ppf.solve(
            problem=two_obj_problem,
            seed=0,
        )
        assert objectives is not None
        cast(MOGD, ppf.solver).patience = 100
        np.testing.assert_array_equal(objectives, np.array([[0, 0]]))
        assert variables is not None
        assert variables[0] == {"v1": 0.0, "v2": 1.0}

    def test_solve_with_three_objectives(
        self, ppf: ParallelProgressiveFrontier, three_obj_problem: MOProblem
    ) -> None:
        set_deterministic_torch()
        obj_values, var_values = ppf.solve(problem=three_obj_problem, seed=0)
        assert obj_values is not None
        np.testing.assert_array_almost_equal(obj_values, np.array([[-1.0, -1.0, -2.0]]))
        assert var_values[0] == {"v1": 1.0, "v2": 7.0}
