from abc import ABC, abstractmethod
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch as th

from ..concepts import SOProblem, Variable
from ..utils.exceptions import NoSolutionError
from .so_solver import SOSolver


class SamplerSolver(SOSolver, ABC):
    def __init__(
        self,
        device: Optional[th.device] = None,
    ) -> None:
        super().__init__()

        self.device = device or th.device("cpu")

    @abstractmethod
    def _get_input(
        self, variables: Mapping[str, Variable], seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        generate samples of variables

        Parameters
        -----------
        variables: List[Variable],
            lower and upper var_ranges of variables(non-ENUM),
            and values of ENUM variables
        seed: int, by default None
            random seed

        Returns
        --------
        np.ndarray,
            variables (n_samples * n_vars)
        """
        pass

    def solve(
        self, problem: SOProblem, seed: Optional[int] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Solve a single-objective optimization problem

        Parameters
        ----------
        objective : Objective
            Objective to be optimized
        variables : Dict[str, Variable]
            Variables to be optimized
        constraints : Optional[List[Constraint]], optional
            List of constraints to comply with, by default None
        input_parameters : Optional[Dict[str, Any]], optional
            Fixed parameters input to objectives and/or constraints,
            by default None
        seed: int, by default None
            random seed
        Returns
        -------
        Point
            A point that satisfies the constraints
            and optimizes the objective

        Raises
        ------
        NoSolutionError
            If no feasible solution is found
        """
        if self.device:
            for constraint in problem.constraints:
                constraint.to(self.device)
            problem.objective.to(self.device)

        if problem.constraints is None:
            pass
        variable_values = self._get_input(problem.variables, seed=seed)
        filtered_vars = self.filter_on_constraints(
            input_vars=variable_values,
            problem=problem,
        )
        if any([len(v) == 0 for v in filtered_vars.values()]):
            raise NoSolutionError("No feasible solution found!")

        th_value = problem.apply_function(
            optimization_element=problem.objective,
            input_variables=filtered_vars,
            device=self.device,
        )
        objective_value = th_value.detach().cpu().numpy().reshape(-1, 1)
        op_ind = int(np.argmin(objective_value * problem.objective.direction))

        return (
            objective_value[op_ind],
            {k: v[op_ind] for k, v in filtered_vars.items()},
        )

    def filter_on_constraints(
        self,
        input_vars: Dict[str, np.ndarray],
        problem: SOProblem,
    ) -> Dict[str, np.ndarray]:
        """Keep only input variables that don't violate constraints

        Parameters
        ----------
        wl_id : str | None
            workload id
        input_vars : np.ndarray
            input variables
        constraints : List[Constraint]
            constraint functions
        """
        if not problem.constraints:
            return input_vars

        available_indices = np.arange(len(next(iter(input_vars.values()))))
        for constraint in problem.constraints:
            const_values = (
                problem.apply_function(constraint, input_vars, device=self.device)
                .detach()
                .cpu()
                .numpy()
            )

            if constraint.upper is not None:
                available_indices = np.intersect1d(
                    available_indices, np.where(const_values <= constraint.upper)
                )
            if constraint.lower is not None:
                available_indices = np.intersect1d(
                    available_indices, np.where(const_values >= constraint.lower)
                )
        filtered_vars = {k: v[available_indices] for k, v in input_vars.items()}
        return filtered_vars
