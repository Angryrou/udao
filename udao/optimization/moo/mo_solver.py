from abc import ABC, abstractmethod
from typing import Any, Optional

from ..concepts.problem import MOProblem


class MOSolver(ABC):
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def solve(
        self,
        problem: MOProblem,
        seed: Optional[int] = None,
    ) -> Any:
        """_summary_

        Parameters
        ----------
        problem : MOProblem
            Multi-objective optimization problem to solve
        seed : Optional[int], optional
            Random seed, by default None

        Returns
        -------
        Any
            A tuple of the objectives values and the variables
            that optimize the objective
        """
        ...
