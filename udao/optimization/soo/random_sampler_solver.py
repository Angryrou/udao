from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np
import torch as th

from ..concepts import EnumVariable, FloatVariable, IntegerVariable, Variable
from ..utils.moo_utils import get_default_device
from .sampler_solver import SamplerSolver


class RandomSamplerSolver(SamplerSolver):
    """Solving a SOO problem by random sampling over variables"""

    @dataclass
    class Params:
        n_samples_per_param: int
        "the number of samples per variable"
        device: Optional[th.device] = field(default_factory=get_default_device)
        """device on which to perform torch operations, by default available device."""

    def __init__(self, params: Params) -> None:
        super().__init__(params.device)
        self.n_samples_per_param = params.n_samples_per_param

    def _process_variable(
        self, var: Variable, seed: Optional[int] = None
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        """Generate samples of a variable"""
        if isinstance(var, FloatVariable):
            return np.random.uniform(var.lower, var.upper, self.n_samples_per_param)
        elif isinstance(var, IntegerVariable):
            return np.random.randint(
                var.lower, var.upper + 1, size=self.n_samples_per_param
            )
        elif isinstance(var, EnumVariable):
            inds = np.random.randint(0, len(var.values), size=self.n_samples_per_param)
            return np.array(var.values)[inds]
        else:
            raise NotImplementedError(
                f"ERROR: variable type {type(var)} is not supported!"
            )

    def _get_input(
        self, variables: Mapping[str, Variable], seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        generate samples of variables

        Parameters:
        -----------
        variables: List[Variable],
            lower and upper var_ranges of variables(non-ENUM),
            and values of ENUM variables
        Returns:
        --------
        Dict[str, np.ndarray]
            Dict with array of values for each variable
        """
        result_dict = {}

        for name, var in variables.items():
            result_dict[name] = self._process_variable(var, seed=seed)

        return result_dict
