from typing import cast

import numpy as np
import pytest
import torch as th

from ....model.utils.utils import set_deterministic_torch
from ....utils.interfaces import VarTypes
from ...concepts.problem import MOProblem
from ...moo.progressive_frontier import SequentialProgressiveFrontier
from ...soo.mogd import MOGD
from ...utils.moo_utils import Point


@pytest.fixture
def spf(mogd: MOGD) -> SequentialProgressiveFrontier:
    spf = SequentialProgressiveFrontier(
        params=SequentialProgressiveFrontier.Params(),
        solver=mogd,
    )
    return spf


class TestProgressiveFrontier:
    def test__get_corner_points(self, spf: SequentialProgressiveFrontier) -> None:
        utopia = Point(np.array([1, 0.3]))
        nadir = Point(np.array([5, 10]))
        corner_points = spf._get_corner_points(utopia, nadir)
        # 1-------3#
        #         #
        # 0-------2#
        expected_points = [
            Point(np.array([1.0, 0.3])),
            Point(np.array([1.0, 10.0])),
            Point(np.array([5.0, 0.3])),
            Point(np.array([5.0, 10.0])),
        ]
        assert all(c == e for c, e in zip(corner_points, expected_points))

    def test__generate_sub_rectangles_bad(
        self, spf: SequentialProgressiveFrontier
    ) -> None:
        utopia = Point(np.array([1, 0.3]))
        nadir = Point(np.array([5, 10]))
        middle = Point((utopia.objs + nadir.objs) / 2)

        rectangles = spf.generate_sub_rectangles(
            utopia, nadir, middle, successful=False
        )
        ############
        #  0 |  1  #
        ############
        #  - |  -  #
        ############
        assert len(rectangles) == 2
        assert rectangles[0].utopia == Point(np.array([1.0, 5.15]))
        assert rectangles[0].nadir == Point(np.array([3.0, 10]))
        assert rectangles[1].utopia == Point(np.array([3.0, 5.15]))
        assert rectangles[1].nadir == Point(np.array([5.0, 10]))

    def test__generate_sub_rectangles_good(
        self, spf: SequentialProgressiveFrontier
    ) -> None:
        utopia = Point(np.array([1, 0.3]))
        nadir = Point(np.array([5, 10]))
        middle = Point((utopia.objs + nadir.objs) / 2)

        rectangles = spf.generate_sub_rectangles(utopia, nadir, middle)
        ############
        #  1 |  _  #
        ############
        #  0 |  2  #
        ############
        assert len(rectangles) == 3
        assert rectangles[0].utopia == Point(np.array([1.0, 0.3]))
        assert rectangles[0].nadir == Point(np.array([3.0, 5.15]))
        assert rectangles[1].utopia == Point(np.array([1.0, 5.15]))
        assert rectangles[1].nadir == Point(np.array([3.0, 10.0]))
        assert rectangles[2].utopia == Point(np.array([3.0, 0.3]))
        assert rectangles[2].nadir == Point(np.array([5.0, 5.15]))

    def test_get_utopia_and_nadir(self, spf: SequentialProgressiveFrontier) -> None:
        points = [
            Point(np.array([1, 5]), {"v1": 0.2, "v2": 1}),
            Point(np.array([3, 10]), {"v1": 0.8, "v2": 6}),
            Point(np.array([5, 0.3]), {"v1": 0.5, "v2": 3}),
        ]
        utopia, nadir = spf.get_utopia_and_nadir(points)
        np.testing.assert_array_equal(utopia.objs, np.array([1, 0.3]))
        np.testing.assert_array_equal(nadir.objs, np.array([5, 10]))

    def test_solve(
        self,
        spf: SequentialProgressiveFrontier,
        two_obj_problem: MOProblem,
    ) -> None:
        cast(MOGD, spf.solver).patience = 100
        objectives, variables = spf.solve(
            problem=two_obj_problem,
            seed=0,
        )
        assert objectives is not None
        np.testing.assert_array_equal(objectives, np.array([[0, 0]]))
        assert variables[0] == {"v1": 0.0, "v2": 1.0}

    def test_solve_gpu(
        self,
        spf: SequentialProgressiveFrontier,
        two_obj_problem: MOProblem,
    ) -> None:
        if not th.cuda.is_available():
            pytest.skip("Skip GPU test")
        cast(MOGD, spf.solver).device = th.device("cuda")
        cast(MOGD, spf.solver).patience = 100
        objectives, variables = spf.solve(
            problem=two_obj_problem,
            seed=0,
        )
        assert objectives is not None
        np.testing.assert_array_equal(objectives, np.array([[0, 0]]))
        assert variables[0] == {"v1": 0.0, "v2": 1.0}

    def test_solve_non_processed_problem(
        self,
        spf: SequentialProgressiveFrontier,
        simple_problem: MOProblem,
    ) -> None:
        cast(MOGD, spf.solver).patience = 100
        objectives, variables = spf.solve(
            problem=simple_problem,
            seed=0,
        )
        assert objectives is not None
        np.testing.assert_array_almost_equal(objectives, np.array([[1, 1.3]]))
        assert variables[0] == {"v1": 0.0, "v2": 3.0}

    def test_solve_non_processed_problem_gpu(
        self,
        spf: SequentialProgressiveFrontier,
        simple_problem: MOProblem,
    ) -> None:
        if not th.cuda.is_available():
            pytest.skip("Skip GPU test")

        cast(MOGD, spf.solver).device = th.device("cuda")
        cast(MOGD, spf.solver).patience = 100
        objectives, variables = spf.solve(
            problem=simple_problem,
            seed=0,
        )
        assert objectives is not None
        np.testing.assert_array_almost_equal(objectives, np.array([[1, 1.3]]))
        assert variables[0] == {"v1": 0.0, "v2": 3.0}

    def test_get_utopia_and_nadir_raises_when_no_points(
        self, spf: SequentialProgressiveFrontier
    ) -> None:
        with pytest.raises(ValueError):
            spf.get_utopia_and_nadir([])

    def test_get_utopia_and_nadir_raises_when_inconsistent_points(
        self, spf: SequentialProgressiveFrontier
    ) -> None:
        with pytest.raises(Exception):
            spf.get_utopia_and_nadir(
                [
                    Point(np.array([1, 5]), {"v1": 0.2, "v2": 1}),
                    Point(np.array([3, 10]), {"v1": 0.8, "v2": 6}),
                    Point(np.array([5]), {"v1": 0.5, "v2": 3}),
                ]
            )

    def test_get_anchor_points(
        self,
        spf: SequentialProgressiveFrontier,
        two_obj_problem: MOProblem,
    ) -> None:
        set_deterministic_torch()
        anchor_point = spf.get_anchor_point(
            problem=two_obj_problem,
            obj_ind=0,
            seed=0,
        )
        np.testing.assert_array_almost_equal(
            anchor_point.objs, np.array([0.0, 0.6944444])
        )
        assert anchor_point.vars == {"v1": 0.0, "v2": 6.0}
        anchor_point = spf.get_anchor_point(
            problem=two_obj_problem,
            obj_ind=1,
            seed=0,
        )
        np.testing.assert_array_almost_equal(
            anchor_point.objs, np.array([0.301196, 0.0])
        )
        assert anchor_point.vars is not None
        assert anchor_point.vars == {"v1": 0.54881352186203, "v2": 1.0}

    def test_get_anchor_points_with_int(
        self,
        spf: SequentialProgressiveFrontier,
        two_obj_problem: MOProblem,
    ) -> None:
        two_obj_problem.objectives[0].type = VarTypes.INT
        set_deterministic_torch()
        anchor_point = spf.get_anchor_point(problem=two_obj_problem, obj_ind=0, seed=0)
        np.testing.assert_array_almost_equal(anchor_point.objs, np.array([0.0, 0.0]))
        assert anchor_point.vars == {"v1": 0.0, "v2": 1.0}
