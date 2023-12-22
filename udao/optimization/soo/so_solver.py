from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from ..concepts import SOProblem


class SOSolver(ABC):
    @abstractmethod
    def solve(
        self,
        problem: SOProblem,
        seed: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Solve a single-objective optimization problem

        Parameters
        ----------
        problem : SOProblem
            Single-objective optimization problem to solve
        seed : Optional[int], optional
            Random seed, by default None

        Returns
        -------
        Tuple[float, Dict[str, float]]
            A tuple of the objective value and the variables
            that optimize the objective
        """
        ...
