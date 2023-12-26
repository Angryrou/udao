from typing import Any, Callable, Optional, Union

import torch as th

from .utils import UdaoFunction


class Constraint:
    """An optimization element is either an objective or a constraint.

    The choice of the type depends on whether a DataProcessor is specified
    for the problem:
    - if no DataProcessor is provided: UdaoFunction, it is a callable
    that takes input_variables and input_parameters
    - else, th.nn.Module or other Callable returning a tensor.

    Parameters
    ----------
    function : Union[UdaoFunction, th.nn.Module, Callable[..., th.Tensor]]
        Objective function, either a UdaoFunction
        or a th.nn.Module if a DataProcessor is provided
    lower : Optional[float], optional
        lower bound of the element, by default None
    upper : Optional[float], optional
        upper bound of the element, by default None
    """

    def __init__(
        self,
        function: Union[UdaoFunction, th.nn.Module, Callable[..., th.Tensor]],
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> None:
        if isinstance(function, th.nn.Module):
            function.eval()
            for p in function.parameters():
                p.requires_grad = False
        self.function = function
        self.lower = lower
        self.upper = upper

    def __call__(self, *args: Any, **kwargs: Any) -> th.Tensor:
        return self.function(*args, **kwargs)

    def to(self, device: Optional[th.device]) -> "Constraint":
        if isinstance(self.function, th.nn.Module) and device is not None:
            self.function.to(device)
        return self

    def __repr__(self) -> str:
        return f"Constraint(lower={self.lower}, upper={self.upper})"
