from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


@dataclass
class Variable:
    """Variable to optimize."""

    pass


@dataclass
class NumericVariable(Variable):
    lower: Union[int, float]
    upper: Union[int, float]

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(
                f"ERROR: the lower bound of variable {self}"
                " is greater than its upper bound!"
            )


@dataclass
class IntegerVariable(NumericVariable):
    """Numeric variable with integer values."""

    lower: int
    upper: int


@dataclass
class FloatVariable(NumericVariable):
    """Numeric variable with float values."""

    lower: float
    upper: float


@dataclass
class BoolVariable(IntegerVariable):
    """Boolean variable."""

    lower: int = field(default=0, init=False)
    upper: int = field(default=1, init=False)


@dataclass
class EnumVariable(Variable):
    """Categorical variable (non-numeric)"""

    values: list


def get_random_variable_values(
    var: Variable, n_samples: int, seed: Optional[int] = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    if isinstance(var, FloatVariable):
        return np.random.uniform(var.lower, var.upper, n_samples)
    elif isinstance(var, IntegerVariable):
        return np.random.randint(int(var.lower), int(var.upper + 1), size=n_samples)
    elif isinstance(var, EnumVariable):
        inds = np.random.randint(0, len(var.values), size=n_samples)
        return np.array(var.values)[inds]
    else:
        raise NotImplementedError(f"ERROR: variable type {type(var)} is not supported!")
