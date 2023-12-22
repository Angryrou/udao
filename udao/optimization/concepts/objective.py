from typing import Callable, Optional, Union

import torch as th

from ...utils.interfaces import VarTypes
from ..concepts.utils import UdaoFunction
from .constraint import Constraint


class Objective(Constraint):
    """

    Parameters
    ----------
    name : str
        Name of the objective.
    minimize : bool
        Direction of the objective: if True, minimize, else maximize.
    type: VarTypes
        Type of the objective, by default VarTypes.FLOAT
    """

    def __init__(
        self,
        name: str,
        minimize: bool,
        function: Union[UdaoFunction, th.nn.Module, Callable[..., th.Tensor]],
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        type: VarTypes = VarTypes.FLOAT,
    ):
        super().__init__(function=function, lower=lower, upper=upper)
        self.name = name
        self.minimize = minimize
        self.type = type

    @property
    def direction(self) -> int:
        """Get gradient direction from optimization type"""
        if self.minimize:
            return 1
        else:
            return -1

    def __repr__(self) -> str:
        return (
            f"Objective(name={self.name}, "
            f"direction={'min' if self.minimize else 'max'}, "
            f"lower={self.lower}, upper={self.upper})"
        )
