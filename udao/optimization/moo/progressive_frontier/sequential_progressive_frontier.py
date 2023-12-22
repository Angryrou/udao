import heapq
import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ....utils.logging import logger
from ...concepts.problem import MOProblem
from ...soo.so_solver import SOSolver
from ...utils import moo_utils as moo_ut
from ...utils.exceptions import NoSolutionError
from ...utils.moo_utils import Point, Rectangle
from .base_progressive_frontier import BaseProgressiveFrontier


class SequentialProgressiveFrontier(BaseProgressiveFrontier):
    """
    Sequential Progressive Frontier -
    a progressive frontier algorithm that explores the uncertainty space
    sequentially, by dividing the space into subrectangles and
    exploring the subrectangles one by one.
    """

    @dataclass
    class Params(BaseProgressiveFrontier.Params):
        n_probes: int = 10
        """number of probes"""

    def __init__(
        self,
        solver: SOSolver,
        params: Params,
    ) -> None:
        super().__init__(solver, params)
        self.n_probes = params.n_probes

    def solve(
        self,
        problem: MOProblem,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve MOO by Progressive Frontier

        Parameters
        ----------
        problem : MOProblem
            MOO problem to be solved

        Returns
        -------
        Tuple[np.ndarray | None, np.ndarray | None]
            optimal objectives and variables
            None, None if no solution is found
        """
        rectangle_queue: List[Rectangle] = []
        n_objs = len(problem.objectives)
        plans = [
            self.get_anchor_point(problem=problem, obj_ind=i, seed=seed)
            for i in range(n_objs)
        ]
        utopia, nadir = self.get_utopia_and_nadir(plans)
        rectangle = Rectangle(utopia, nadir)
        heapq.heappush(rectangle_queue, rectangle)
        for _ in range(self.n_probes - n_objs):
            if not rectangle_queue:
                logger.info("No more uncertainty space to explore further!")
                break
            rectangle = heapq.heappop(rectangle_queue)
            middle_point, subrectangles = self._find_local_optimum(
                problem, rectangle, seed=seed
            )
            if middle_point is not None:
                plans.append(middle_point)
            for sub_rect in subrectangles:
                if sub_rect.volume != 0:
                    heapq.heappush(rectangle_queue, sub_rect)

        ## filter dominated points
        po_objs_list = [point.objs.tolist() for point in plans]
        po_vars_list = [point.vars for point in plans]
        po_objs, po_vars = moo_ut.summarize_ret(po_objs_list, po_vars_list)

        return po_objs, po_vars

    def _find_local_optimum(
        self,
        problem: MOProblem,
        rectangle: Rectangle,
        seed: Optional[int] = None,
    ) -> Tuple[Optional[Point], List[Rectangle]]:
        """
        Find the local optimum in the given rectangle and
        the subrectangles in which to continue the search.
        If no optimum is found for the rectangle,
        return None and the upper subrectangles.
        If an optimum is found, return the optimum and
        the subrectangles in which to continue the search.

        Parameters
        ----------
        rectangle : Rectangle
            Rectangle in which to find the local optimum
        wl_id : str | None
            workload id

        Returns
        -------
        Tuple[Point | None, List[Rectangle]]
            The local optimum and the subrectangles in which to continue the search
        """
        current_utopia, current_nadir = Point(rectangle.lower_bounds), Point(
            rectangle.upper_bounds
        )
        logger.debug(f"current utopia is: {current_utopia}")
        logger.debug(f"current nadir is: {current_nadir}")
        middle_objs = np.array(
            [
                current_nadir.objs[i]
                if i == self.opt_obj_ind
                else (current_utopia.objs[i] + current_nadir.objs[i]) / 2
                for i in range(len(problem.objectives))
            ]
        )
        middle_point = Point(middle_objs)
        obj_bounds_dict = self._form_obj_bounds_dict(
            problem, current_utopia, middle_point
        )
        logger.debug(f"obj_bounds are: {obj_bounds_dict}")
        so_problem = self._so_problem_from_bounds_dict(
            problem, obj_bounds_dict, problem.objectives[self.opt_obj_ind]
        )
        try:
            _, soo_vars = self.solver.solve(so_problem, seed=seed)
        except NoSolutionError:
            logger.debug(
                "This is an empty area \n "
                "don't have pareto points, only "
                "divide current uncertainty space"
            )
            middle_point = Point((current_utopia.objs + current_nadir.objs) / 2)
            rectangles = self.generate_sub_rectangles(
                current_utopia, current_nadir, middle_point, successful=False
            )
            return None, rectangles
        else:
            middle_objs = self._compute_objectives(problem, soo_vars)
            middle_point = Point(middle_objs, soo_vars)
            return middle_point, self.generate_sub_rectangles(
                current_utopia, current_nadir, middle_point
            )

    def generate_sub_rectangles(
        self, utopia: Point, nadir: Point, middle: Point, successful: bool = True
    ) -> List[Rectangle]:
        """

        Generate uncertainty space to be explored:
        - if starting from successful optimum as middle, excludes the dominated
        space (middle as utopia and nadir as nadir)
        - if starting from unsuccessful optimum as middle, excludes the space where
        all constraining objectives are lower than the middle point.

        Parameters
        ----------
        utopia: Point
            the utopia point
        nadir: Point
            the nadir point
        middle: Point
            the middle point generated by
            the constrained single objective optimization
        successful: bool
            whether the middle point is from a successful optimization

        Returns
        -------
        List[Rectangle]
            sub rectangles to be explored
        """

        rectangles = []
        corner_points = self._get_corner_points(utopia, nadir)
        for point in corner_points:
            # space explored (lower half of constraining objectives)
            is_explored_unconclusive = not successful and np.all(
                middle.objs[1:] - point.objs[1:] > 0
            )
            # nadir point
            is_dominated = successful and np.all(middle.objs - point.objs < 0)
            if is_dominated or is_explored_unconclusive:
                continue
            sub_rect_u, sub_rect_n = self.get_utopia_and_nadir([point, middle])
            rectangles.append(Rectangle(sub_rect_u, sub_rect_n))

        return rectangles

    def _get_corner_points(self, utopia: Point, nadir: Point) -> List[Point]:
        """
        get the corner points that can form a hyper_rectangle
        from utopia and nadir points.

        Parameters
        ----------
        utopia: Points (defined by class), the utopia point
        nadir: Points (defined by class), the nadir point

        Returns
        -------
        List[Point]
            2^n_objs corner points
        """
        n_objs = utopia.n_objs
        u_obj_values, n_obj_values = utopia.objs.reshape(
            [n_objs, 1]
        ), nadir.objs.reshape([n_objs, 1])
        grids_list = np.hstack([u_obj_values, n_obj_values])

        ## generate cartesian product of grids_list
        objs_corner_points = np.array([list(i) for i in itertools.product(*grids_list)])
        corner_points = [Point(objs=obj_values) for obj_values in objs_corner_points]

        return corner_points
