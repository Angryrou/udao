import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch as th

from ..concepts import EnumVariable, IntegerVariable, NumericVariable, Variable
from ..utils.moo_utils import get_default_device
from .sampler_solver import SamplerSolver


class GridSearchSolver(SamplerSolver):
    """Solving a SOO problem by grid search over variables"""

    @dataclass
    class Params:
        n_grids_per_var: List[int]
        """List of grid sizes for each variable"""
        device: Optional[th.device] = field(default_factory=get_default_device)
        """device on which to perform torch operations, by default available device."""

    def __init__(self, params: Params) -> None:
        """
        :param gs_params: dict, the parameters used in grid_search
        """
        super().__init__(params.device)
        self.n_grids_per_var = params.n_grids_per_var

    def _process_variable(self, var: Variable, n_grids: int) -> np.ndarray:
        """Define grid point in fonction of the variable type"""
        if isinstance(var, NumericVariable):
            # make sure the grid point is the same with the type
            # e.g., if int x.min=0, x.max=5, n_grids_per_var=10,
            # ONLY points[0, 1, 2, 3, 4, 5] are feasible
            if isinstance(var, IntegerVariable):
                if n_grids > (var.upper - var.lower + 1):
                    n_grids = int(var.upper - var.lower + 1)

            var_grid = np.linspace(var.lower, var.upper, num=n_grids, endpoint=True)
            if isinstance(var, IntegerVariable):
                return np.round(var_grid).astype(int)
            return var_grid
        elif isinstance(var, EnumVariable):
            return np.array(var.values)
        else:
            raise NotImplementedError(
                f"ERROR: variable type {type(var)} is not supported!"
            )

    def _get_input(
        self, variables: Mapping[str, Variable], seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate grids for each variable

        Parameters
        ----------
        variables: Mapping[str, Variable]
            variables to generate

        Returns
        -------
        Dict[str, np.ndarray]
            Dict with array of values for each variable
        """
        grids_list = []
        variable_names = list(variables.keys())

        for i, var_name in enumerate(variable_names):
            var = variables[var_name]
            var_n_grids = self.n_grids_per_var[i]
            grids_list.append({var_name: self._process_variable(var, var_n_grids)})

        values_list = [list(d.values())[0] for d in grids_list]
        cartesian_product = np.array([list(i) for i in itertools.product(*values_list)])
        result_dict = {
            var_name: cartesian_product.T[i]
            for i, var_name in enumerate(variable_names)
        }

        return result_dict
