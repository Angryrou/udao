import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ..concepts import Objective
from ..concepts.problem import MOProblem
from ..soo.mogd import MOGD
from ..soo.so_solver import SOSolver
from ..utils import moo_utils as moo_ut
from ..utils.exceptions import NoSolutionError
from ..utils.moo_utils import Point, get_default_device
from .mo_solver import MOSolver


class WeightedSumObjective(Objective):
    """Weighted Sum Objective"""

    def __init__(
        self,
        problem: MOProblem,
        ws: List[float],
        allow_cache: bool = False,
        normalize: bool = True,
        device: Optional[th.device] = None,
    ) -> None:
        self.device = device or get_default_device()
        self.problem = problem
        self.ws = ws
        super().__init__(name="weighted_sum", function=self.function, minimize=True)
        self._cache: Dict[str, th.Tensor] = {}
        self.allow_cache = allow_cache
        self.normalize = normalize

    def _function(self, *args: Any, **kwargs: Any) -> th.Tensor:
        hash_var = ""
        if self.allow_cache:
            hash_var = json.dumps(str(args) + str(kwargs))
            if hash_var in self._cache:
                return self._cache[hash_var]
        objs: List[th.Tensor] = []
        for objective in self.problem.objectives:
            obj = objective(*args, **kwargs) * objective.direction
            objs.append(obj.squeeze())
        objs_tensor = th.vstack(objs).T
        # shape (n_feasible_samples/grids, n_objs)
        if self.allow_cache:
            self._cache[hash_var] = objs_tensor
        return objs_tensor

    def function(self, *args: Any, **kwargs: Any) -> th.Tensor:
        """Sum of weighted normalized objectives"""
        objs_tensor = self._function(*args, **kwargs)
        if self.normalize:
            objs_tensor = self._normalize_objective(objs_tensor)
        return th.sum(objs_tensor * th.tensor(self.ws, device=self.device), dim=1)

    def _normalize_objective(self, objs_array: th.Tensor) -> th.Tensor:
        """Normalize objective values to [0, 1]

        Parameters
        ----------
        objs_array : np.ndarray
            shape (n_feasible_samples/grids, n_objs)

        Returns
        -------
        np.ndarray
            shape (n_feasible_samples/grids, n_objs)

        Raises
        ------
        NoSolutionError
            if lower bounds of objective values are
            higher than their upper bounds
        """
        objs_min, objs_max = th.min(objs_array, 0).values, th.max(objs_array, 0).values
        if th.any((objs_min - objs_max) > 0):
            raise NoSolutionError(
                "Cannot do normalization! Lower bounds of "
                "objective values are higher than their upper bounds."
            )
        elif th.all((objs_min - objs_max) == 0):
            return th.zeros_like(objs_array)
        return (objs_array - objs_min) / (objs_max - objs_min)

    def to(self, device: Optional[th.device] = None) -> "WeightedSumObjective":
        """Move objective to device"""
        if device is None:
            device = get_default_device()

        self.device = device
        for objective in self.problem.objectives:
            objective.to(device)
        for constraint in self.problem.constraints:
            constraint.to(device)
        self._cache = {k: v.to(device) for k, v in self._cache.items()}
        return self


class WeightedSum(MOSolver):
    """
    Weighted Sum (WS) algorithm for MOO

    Parameters
    ----------
    ws_pairs: np.ndarray,
        weight settings for all objectives, of shape (n_weights, n_objs)
    inner_solver: BaseSolver,
        the solver used in Weighted Sum
    objectives: List[Objective],
        objective functions
    constraints: List[Constraint],
        constraint functions

    """

    @dataclass
    class Params:
        ws_pairs: np.ndarray
        """weight sets for all objectives, of shape (n_weights, n_objs)"""
        so_solver: SOSolver
        """solver for SOO"""
        normalize: bool = True
        """whether to normalize objective values to [0, 1] before applying WS"""
        allow_cache: bool = False
        """whether to cache the objective values"""
        device: Optional[th.device] = field(default_factory=get_default_device)
        """device on which to perform torch operations, by default available device."""

    def __init__(
        self,
        params: Params,
    ):
        super().__init__()
        self.so_solver = params.so_solver
        self.ws_pairs = params.ws_pairs
        self.allow_cache = params.allow_cache
        self.normalize = params.normalize
        self.device = params.device

        if self.allow_cache and isinstance(params.so_solver, MOGD):
            raise NotImplementedError(
                "MOGD does not support caching." "Please set allow_cache=False."
            )

    def solve(
        self, problem: MOProblem, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """solve MOO problem by Weighted Sum (WS)

        Parameters
        ----------
        variables : List[Variable]
            List of the variables to be optimized.
        input_parameters : Optional[Dict[str, Any]]
            Fixed input parameters expected by
            the objective functions.

        Returns
        -------
        Tuple[Optional[np.ndarray],Optional[np.ndarray]]
            Pareto solutions and corresponding variables.
        """
        candidate_points: List[Point] = []
        objective = WeightedSumObjective(
            problem, self.ws_pairs[0], self.allow_cache, self.normalize, self.device
        )
        so_problem = problem.derive_SO_problem(objective)
        for i, ws in enumerate(self.ws_pairs):
            objective.ws = ws
            _, soo_vars = self.so_solver.solve(
                so_problem,
                seed=seed + i * (not self.allow_cache) if seed is not None else None,
            )

            objective_values = np.array(
                [
                    problem.apply_function(obj, soo_vars, device=self.device)
                    .cpu()
                    .numpy()
                    for obj in problem.objectives
                ]
            ).T.squeeze()

            candidate_points.append(Point(objective_values, soo_vars))

        return moo_ut.summarize_ret(
            [point.objs for point in candidate_points],
            [point.vars for point in candidate_points],
        )
