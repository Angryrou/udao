import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th
from torch.multiprocessing import Pool

from ....utils.logging import logger
from ...concepts import MOProblem, Objective, SOProblem
from ...soo.so_solver import SOSolver
from ...utils import moo_utils as moo_ut
from ...utils.exceptions import NoSolutionError
from ...utils.moo_utils import Point, Rectangle
from .base_progressive_frontier import BaseProgressiveFrontier


class ParallelProgressiveFrontier(BaseProgressiveFrontier):
    @dataclass
    class Params(BaseProgressiveFrontier.Params):
        processes: int = 1
        """Processes to use for parallel processing"""
        n_grids: int = 2
        """Number of splits per objective"""
        max_iters: int = 10
        """Number of iterations to explore the space"""

    def __init__(
        self,
        solver: SOSolver,
        params: Params,
    ) -> None:
        super().__init__(
            solver,
            params,
        )
        self.processes = params.processes
        self.n_grids = params.n_grids
        self.max_iters = params.max_iters

    def solve(
        self,
        problem: MOProblem,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        solve MOO by PF-AP (Progressive Frontier - Approximation Parallel)

        Parameters
        ----------
        problem : MOProblem
            MOO problem to be solved
        Returns
        -------
        po_objs: ndarray
            Pareto optimal objective values, of shape
            (n_solutions, n_objs)
        po_vars: ndarray
            corresponding variables of Pareto solutions, of shape
            (n_solutions, n_vars)
        """
        # create initial rectangle
        # get initial plans/form a intial hyperrectangle

        plans: List[Point] = []
        n_objs = len(problem.objectives)

        all_objs_list: List[np.ndarray] = []
        all_vars_list: List[Dict] = []
        for i in range(n_objs):
            anchor_point = self.get_anchor_point(problem=problem, obj_ind=i, seed=seed)
            if anchor_point.vars is None:
                raise Exception("This should not happen.")
            plans.append(anchor_point)
            all_objs_list.append(anchor_point.objs)
            all_vars_list.append(anchor_point.vars)
        logger.debug(f"the initial plans are: {plans}")
        if n_objs < 2 or n_objs > 3:
            raise Exception(f"{n_objs} objectives are not supported for now!")

        for i in range(self.max_iters):
            # choose the cell with max volume to explore
            max_volume = -1
            input_ind = -1
            for i in range(len(all_objs_list) - 1):
                current_volume = abs(
                    np.prod(np.array(all_objs_list)[i] - np.array(all_objs_list)[i + 1])
                )
                logger.debug(f"volume {current_volume}")
                if current_volume > max_volume:
                    max_volume = current_volume
                    input_ind = i

            plan = [
                Point(objs=np.array(all_objs_list)[input_ind]),
                Point(objs=np.array(all_objs_list)[input_ind + 1]),
            ]
            utopia, nadir = self.get_utopia_and_nadir(plan)
            if utopia is None or nadir is None:
                raise NoSolutionError("Cannot find utopia/nadir points")
            # create uniform n_grids ^ (n_objs) grid cells based on the rectangle
            grid_cells_list = self._create_grid_cells(
                utopia, nadir, self.n_grids, n_objs
            )

            obj_bound_cells = []
            for cell in grid_cells_list:
                obj_bound_dict = self._form_obj_bounds_dict(
                    problem, cell.utopia, cell.nadir
                )
                obj_bound_cells.append(obj_bound_dict)

            logger.debug(f"the cells are: {obj_bound_cells}")
            ret_list = self.parallel_soo(
                problem=problem,
                objective=problem.objectives[self.opt_obj_ind],
                cell_list=obj_bound_cells,
                seed=seed,
            )

            po_objs_list: List[np.ndarray] = []
            po_vars_list: List[Dict] = []
            for soo_obj, soo_vars in ret_list:
                if soo_obj is None:
                    logger.debug("This is an empty area!")
                    continue
                else:
                    po_objs_list.append(self._compute_objectives(problem, soo_vars))
                    if soo_vars is None:
                        raise Exception("Unexpected vars None for objective value.")
                    po_vars_list.append(soo_vars)

            logger.debug(f"the po_objs_list is: {po_objs_list}")
            logger.debug(f"the po_vars_list is: {po_vars_list}")
            all_objs_list.extend(po_objs_list)
            all_vars_list.extend(po_vars_list)
            logger.debug(f"the all_objs_list is: {all_objs_list}")
            all_objs, all_vars = moo_ut.summarize_ret(all_objs_list, all_vars_list)
            all_objs_list = all_objs.tolist() if all_objs is not None else []
            all_vars_list = all_vars.tolist() if all_vars is not None else []

        return np.array(all_objs_list), np.array(all_vars_list)

    def _solve_wrapper(
        self, problem: SOProblem, seed: Optional[int] = None
    ) -> Optional[Tuple[float, Dict[str, Any]]]:
        """Handle exceptions in solver call for parallel processing."""
        try:
            return self.solver.solve(problem, seed=seed)
        except NoSolutionError:
            logger.debug(f"This is an empty area! {problem}")
            return None

    def parallel_soo(
        self,
        problem: MOProblem,
        objective: Objective,
        cell_list: List[Dict[str, Any]],
        seed: Optional[int] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Parallel calls to SOO Solver for each cell in cell_list, returns a
        candidate tuple (objective_value, variables) for each cell.

        Parameters
        ----------
        objective : Objective
            Objective to be optimized
        cell_list : List[Dict[str, Any]]
            List of cells to be optimized
            (a cell is a dict of bounds for each objective)
        input_parameters : Optional[Dict[str, Any]], optional
            Fixed parameters to be passed , by default None

        Returns
        -------
        List[Tuple[float, Dict[str, Any]]]
            List of candidate tuples (objective_value, variables)
        """
        # generate the list of input parameters for constraint_so_opt
        args_list: List[Tuple[SOProblem, Optional[int]]] = []
        for obj_bounds_dict in cell_list:
            so_problem = self._so_problem_from_bounds_dict(
                problem, obj_bounds_dict, objective
            )
            args_list.append((so_problem, seed))

        if th.cuda.is_available():
            th.multiprocessing.set_start_method("spawn", force=True)
        if self.processes == 1:
            ret_list = [self._solve_wrapper(*args) for args in args_list]
        else:
            # call self.constraint_so_opt parallely
            with Pool(processes=self.processes) as pool:
                ret_list = pool.starmap(self._solve_wrapper, args_list)
        return [res for res in ret_list if res is not None]

    @staticmethod
    def _create_grid_cells(
        utopia: Point, nadir: Point, n_grids: int, n_objs: int
    ) -> List[Rectangle]:
        """
        Create cells used in Progressive Frontier(PF)-Approximation
        Parallel (AP) algorithm

        Parameters
        ----------
        utopia: Point
            the utopia point
        nadir: Point
            the nadir point
        n_grids: int
            the number of grids per objective
        n_objs: int
            the number of objectives

        Returns
        -------
            List[Rectangle]
            The rectangles in which to perform optimization.
        """
        grids_per_var = np.linspace(
            utopia.objs, nadir.objs, num=n_grids + 1, endpoint=True
        )
        objs_list = [grids_per_var[:, i] for i in range(n_objs)]

        ## generate cartesian product of indices for grids
        grids_inds_per_var = np.linspace(0, n_grids - 1, num=n_grids, endpoint=True)
        x = np.tile(grids_inds_per_var, (n_objs, 1))
        grids_inds = np.array([list(i) for i in itertools.product(*x)]).astype(int)

        grid_cell_list = []
        for grid_ind in grids_inds:
            sub_u_objs = np.array([objs_list[i][id] for i, id in enumerate(grid_ind)])
            sub_u_point = Point(sub_u_objs)
            sub_nadir_objs = np.array(
                [objs_list[i][id + 1] for i, id in enumerate(grid_ind)]
            )
            sub_nadir_point = Point(sub_nadir_objs)
            assert all((sub_nadir_objs - sub_u_objs) >= 0)
            cell = Rectangle(sub_u_point, sub_nadir_point)
            grid_cell_list.append(cell)
        if len(grid_cell_list) != (n_grids**n_objs):
            raise Exception(
                f"Unexpected: the number of grid cells is"
                f"not equal to {n_grids**n_objs}"
            )

        return grid_cell_list
