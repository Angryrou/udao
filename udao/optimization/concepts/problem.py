from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch as th

from ...data.handler.data_processor import DataProcessor
from ..concepts.utils import (
    InputVariables,
    derive_processed_input,
    derive_unprocessed_input,
)
from .constraint import Constraint
from .objective import Objective
from .variable import Variable


class BaseProblem:
    """Base class for optimization problems."""

    def __init__(
        self,
        variables: Dict[str, Variable],
        constraints: Sequence[Constraint],
        data_processor: Optional[DataProcessor] = None,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        variables : Dict[str, Variable]
            _description_
        constraints : Sequence[Constraint]
            _description_
        data_processor : Optional[DataProcessor], optional
            _description_, by default None
        input_parameters : Optional[Dict[str, Any]], optional
            _description_, by default None
        """
        self.variables = variables
        self.constraints = constraints
        self.data_processor = data_processor
        self.input_parameters = input_parameters

    def apply_function(
        self,
        optimization_element: Constraint,
        input_variables: InputVariables,
        device: Optional[th.device] = None,
    ) -> th.Tensor:
        if self.data_processor is not None:
            input_data, _ = derive_processed_input(
                self.data_processor,
                input_variables=input_variables,
                input_parameters=self.input_parameters,
                device=device,
            )
            th_value = optimization_element.to(device)(input_data)
        else:
            input_vars, input_params = derive_unprocessed_input(
                input_variables,
                self.input_parameters,
                device=device,
            )
            th_value = optimization_element(
                input_parameters=input_params,
                input_variables=input_vars,
            )

        return th_value

    def derive_SO_problem(self, objective: Objective) -> "SOProblem":
        return SOProblem(
            objective=objective,
            variables=self.variables,
            constraints=self.constraints,
            data_processor=self.data_processor,
            input_parameters=self.input_parameters,
        )


@dataclass
class MOProblem(BaseProblem):
    """Multi-objective optimization problem."""

    def __init__(
        self,
        objectives: Sequence[Objective],
        variables: Dict[str, Variable],
        constraints: Sequence[Constraint],
        data_processor: Optional[DataProcessor] = None,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.objectives = objectives
        super().__init__(
            variables,
            constraints,
            data_processor=data_processor,
            input_parameters=input_parameters,
        )

    def __repr__(self) -> str:
        return (
            f"MOProblem(objectives={self.objectives}, "
            f"variables={self.variables}, "
            f"constraints={self.constraints}, "
            f"input_parameters={self.input_parameters})"
        )


@dataclass
class SOProblem(BaseProblem):
    """Single-objective optimization problem."""

    def __init__(
        self,
        objective: Objective,
        variables: Dict[str, Variable],
        constraints: Sequence[Constraint],
        data_processor: Optional[DataProcessor] = None,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.objective = objective
        super().__init__(
            variables,
            constraints,
            data_processor=data_processor,
            input_parameters=input_parameters,
        )

    def __repr__(self) -> str:
        return (
            f"SOProblem(objective={self.objective}, "
            f"variables={self.variables}, "
            f"constraints={self.constraints}, "
            f"input_parameters={self.input_parameters})"
        )
