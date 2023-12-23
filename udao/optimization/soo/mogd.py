from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch as th
import torch.optim as optim

from ...data.containers.tabular_container import TabularContainer
from ...data.handler.data_processor import DataProcessor
from ...data.iterators.base_iterator import UdaoIterator
from ...utils.interfaces import UdaoInput, UdaoItemShape
from ...utils.logging import logger
from .. import concepts as co
from ..concepts.utils import derive_processed_input, derive_unprocessed_input
from ..utils.exceptions import NoSolutionError, UncompliantSolutionError
from .so_solver import SOSolver


def get_default_device() -> th.device:
    return th.device("cuda") if th.cuda.is_available() else th.device("cpu")


class MOGD(SOSolver):
    """MOGD solver for single-objective optimization.

    Performs gradient descent on input variables by minimizing an
    objective loss and a constraint loss.
    """

    @dataclass
    class Params:
        learning_rate: float
        """learning rate of Adam optimizer applied to input variables"""
        max_iters: int
        """maximum number of iterations for a single local search"""
        patience: int
        """maximum number of iterations without improvement"""
        multistart: int
        """number of random starts for gradient descent"""
        objective_stress: float = 10.0
        """stress term for objective functions"""
        constraint_stress: float = 1e5
        """stress term for constraint functions"""
        int_rounding_mode: str = "auto"
        """rounding mode for integer variables, supporting "auto", "all", "once" """
        batch_size: int = 1
        """batch size for gradient descent"""
        device: Optional[th.device] = field(default_factory=get_default_device)
        """device on which to perform torch operations, by default available device."""
        dtype: th.dtype = th.float32
        """type of the tensors"""

    def __init__(self, params: Params) -> None:
        super().__init__()
        self.lr = params.learning_rate
        self.max_iter = params.max_iters
        self.patience = params.patience
        self.multistart = params.multistart
        self.objective_stress = params.objective_stress
        self.constraint_stress = params.constraint_stress
        self.int_rounding_mode = params.int_rounding_mode
        self.batch_size = params.batch_size
        self.device = params.device
        self.dtype = params.dtype

        assert self.int_rounding_mode in [
            "auto",
            "all",
            "once",
        ], f"Invalid rounding mode {self.int_rounding_mode}!"

    def _get_unprocessed_input_values(
        self,
        numeric_variables: Dict[str, co.NumericVariable],
        input_parameters: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, th.Tensor], Dict[str, Any]]:
        """

        Parameters
        ----------
        numeric_variables : Dict[str, co.NumericVariable]
            Numeric variables for which to get random values
        input_parameters : Optional[Dict[str, Any]], optional
            Non decision parts of the input, by default None
        seed : Optional[int], optional
            Random seed, by default None

        Returns
        -------
        Tuple[Dict[str, th.Tensor], Dict[str, Any]]
            - random values as a tensor for each numeric variable
            - input parameters valuies
        """
        numeric_values: Dict[str, np.ndarray] = {}

        for i, (name, variable) in enumerate(numeric_variables.items()):
            numeric_values[name] = co.variable.get_random_variable_values(
                variable, self.batch_size, seed=seed + i if seed is not None else None
            )
        return derive_unprocessed_input(
            input_variables=numeric_values,
            input_parameters=input_parameters,
            device=self.device,
        )

    def _get_processed_input_values(
        self,
        numeric_variables: Dict[str, co.NumericVariable],
        data_processor: DataProcessor,
        input_parameters: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[UdaoInput, UdaoItemShape, Callable[[th.Tensor], TabularContainer],]:
        """Get random values for numeric variables

        Parameters
        ----------
        numeric_variables : Dict[str, co.NumericVariable]
            Numeric variables on which to apply gradients
        data_processor : DataProcessor
            Data processor to process input variables
        input_parameters : Optional[Dict[str, Any]], optional
            Non decision parts of the input, by default None

        Returns
        -------
        Tuple[UdaoInput, UdaoInputShape, Callable[[th.Tensor], TabularContainer]]
            - random values for numeric variables
            - shape of the input
            - function to convert a tensor to a TabularContainer
        """
        numeric_values: Dict[str, np.ndarray] = {}

        for i, (name, variable) in enumerate(numeric_variables.items()):
            numeric_values[name] = co.variable.get_random_variable_values(
                variable, self.batch_size, seed=seed + i if seed is not None else None
            )
        input_data, iterator = derive_processed_input(
            data_processor=data_processor,
            input_parameters=input_parameters or {},
            input_variables=numeric_values,
            device=self.device,
        )
        make_tabular_container = cast(
            UdaoIterator, iterator
        ).get_tabular_features_container

        input_data_shape = iterator.shape

        return (
            input_data,
            input_data_shape,
            make_tabular_container,
        )

    def _get_unprocessed_input_bounds(
        self,
        numeric_variables: Dict[str, co.NumericVariable],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """

        Parameters
        ----------
        numeric_variables : Dict[str, co.NumericVariable]
            Variables for which to get bounds

        Returns
        -------
        Tuple[Dict[str, float], Dict[str, float]]
            - lower bounds of numeric variables
            - upper bounds of numeric variables
        """
        lower_numeric_values = {
            name: variable.lower for name, variable in numeric_variables.items()
        }
        upper_numeric_values = {
            name: variable.upper for name, variable in numeric_variables.items()
        }
        return lower_numeric_values, upper_numeric_values

    def _get_processed_input_bounds(
        self,
        numeric_variables: Dict[str, co.NumericVariable],
        data_processor: DataProcessor,
        input_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[UdaoInput, UdaoInput]:
        """Get bounds of numeric variables

        Parameters
        ----------
        numeric_variables : Dict[str, co.NumericVariable]
            Numeric variables on which to apply gradients
        data_processor : DataProcessor
            Data processor to process input variables
        input_parameters : Optional[Dict[str, Any]], optional
            Input parameters, by default None

        Returns
        -------
        Tuple[UdaoInput, UdaoInput]
            Lower and upper bounds of numeric
            variables in the form of a UdaoInput
        """
        lower_numeric_values = {
            name: variable.lower for name, variable in numeric_variables.items()
        }
        upper_numeric_values = {
            name: variable.upper for name, variable in numeric_variables.items()
        }
        lower_input, _ = derive_processed_input(
            data_processor=data_processor,
            input_parameters=input_parameters,
            input_variables=lower_numeric_values,
        )
        upper_input, _ = derive_processed_input(
            data_processor=data_processor,
            input_parameters=input_parameters,
            input_variables=upper_numeric_values,
        )
        if self.device:
            return lower_input.to(self.device), upper_input.to(self.device)
        else:
            return lower_input, upper_input

    def _gradient_descent(
        self, problem: co.SOProblem, input_data: Any, optimizer: th.optim.Optimizer
    ) -> Tuple[int, float]:
        """Perform a gradient descent step on input variables

        Parameters
        ----------
        problem : co.SOProblem
            Single-objective optimization problem
        input_data : Any
            Input data - can have different types depending on whether
            the input variables are processed or not

        optimizer : th.optim.Optimizer
            PyTorch optimizer

        Returns
        -------
        Tuple[int, float, float]
            - index of minimum loss
            - minimum loss
            - objective value at minimum loss

        Raises
        ------
        UncompliantSolutionError
            If no solution within bounds is found
        """
        # Compute objective, constraints and corresponding losses

        loss = self.compute_loss(problem, input_data)

        sum_loss = th.sum(loss)
        min_loss, min_loss_id = th.min(loss), th.argmin(loss)

        optimizer.zero_grad()
        sum_loss.backward()  # type: ignore
        optimizer.step()
        return (
            int(min_loss_id.item()),
            min_loss.item(),
        )

    def _log_success(
        self,
        problem: co.SOProblem,
        iter: int,
        best_obj: float,
        best_iter: int,
        best_feature_input: Any,
    ) -> None:
        logger.debug(
            f"Finished at iteration {iter}, best local {problem.objective.name} "
            f"found {best_obj:.5f}"
            f" \nat iteration {best_iter},"
            f" \nwith vars: {best_feature_input}, for "
            f"objective {problem.objective} and constraints {problem.constraints}"
        )

    def _log_failure(
        self,
        problem: co.SOProblem,
        iter: int,
    ) -> None:
        logger.debug(
            f"Finished at iteration {iter}, no valid {problem.objective.name}"
            f" found for input parameters {problem.input_parameters} with "
            f"objective {problem.objective} and constraints {problem.constraints}"
        )

    def _unprocessed_single_start_opt(
        self,
        problem: co.SOProblem,
        seed: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float], float]:
        """Perform a single start optimization, in the case where
        no data processor is defined.
        The input variables are transformed to a dictionary of tensors and are
        optimized directly, by being passed to the objective function along
        with the input parameters.
        """
        best_iter = 0
        best_loss = np.inf
        best_obj: Optional[float] = None
        best_raw_vars: Optional[Dict[str, Any]] = None

        (
            input_variable_values,
            input_parameter_values,
        ) = self._get_unprocessed_input_values(
            cast(Dict[str, co.NumericVariable], problem.variables), seed=seed
        )
        lower_input, upper_input = self._get_unprocessed_input_bounds(
            cast(Dict[str, co.NumericVariable], problem.variables)
        )
        for name in input_variable_values:
            input_variable_values[name].requires_grad_(True)
        optimizer = optim.Adam([t for t in input_variable_values.values()], lr=self.lr)
        i = 0
        while i < self.max_iter:
            try:
                min_loss_id, min_loss = self._gradient_descent(
                    problem,
                    {**input_variable_values, **input_parameter_values},
                    optimizer=optimizer,
                )
            except UncompliantSolutionError:
                pass
            else:
                if min_loss < best_loss:
                    raw_vars = self._compute_raw_variables_from_unprocessed_input(
                        problem=problem,
                        input_variable_values={
                            k: v[min_loss_id] for k, v in input_variable_values.items()
                        },
                    )
                    obj_value = problem.apply_function(problem.objective, raw_vars)
                    is_within_objective_bounds = self.within_objective_bounds(
                        obj_value.item(), problem.objective
                    )
                    loss = self.objective_loss(obj_value, problem.objective)
                    is_within_constraints = True
                    if problem.constraints:
                        constraint_loss = self.constraints_loss(
                            [
                                problem.apply_function(constraint, raw_vars)
                                for constraint in problem.constraints
                            ],
                            problem.constraints,
                        )
                        loss += constraint_loss
                        is_within_constraints = bool((constraint_loss == 0).item())
                    if (
                        is_within_objective_bounds
                        and is_within_constraints
                        and loss.item() < best_loss
                    ):
                        best_loss = loss.item()
                        best_obj = obj_value.cpu().item()
                        best_raw_vars = raw_vars

                    best_iter = i
            # Update input_variable_values with constrained values
            with th.no_grad():
                for k in input_variable_values:
                    input_variable_values[k].data = th.clip(
                        input_variable_values[k].data,
                        lower_input[k],
                        upper_input[k],
                    )

            if i > best_iter + self.patience:
                break
            i += 1
        if best_obj is not None:
            self._log_success(problem, i, best_obj, best_iter, best_raw_vars)

        if best_obj is not None and best_raw_vars is not None:
            return best_obj, best_raw_vars, best_loss
        else:
            self._log_failure(problem, i)
            raise NoSolutionError

    def _processed_single_start_opt(
        self,
        problem: co.SOProblem,
        seed: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float], float]:
        """Perform a single start optimization, in the case where
        a data processor is defined.

        input variables and parameters are processed by the data processor.
        Gradient descent is performed on the processed input variables.
        Variables are then inverse transformed to get the raw variables.
        """
        if not problem.data_processor:
            raise Exception("Data processor is not defined!")
        best_iter = 0
        best_loss = np.inf
        best_obj: Optional[float] = None
        best_vars: Optional[Dict[str, Any]] = None
        # Random numeric variables and their characteristics
        (
            input_data,
            input_data_shape,
            make_tabular_container,
        ) = self._get_processed_input_values(
            cast(Dict[str, co.NumericVariable], problem.variables),
            data_processor=problem.data_processor,
            input_parameters=problem.input_parameters,
            seed=seed,
        )
        # Bounds of numeric variables
        lower_input, upper_input = self._get_processed_input_bounds(
            cast(Dict[str, co.NumericVariable], problem.variables),
            data_processor=problem.data_processor,
            input_parameters=problem.input_parameters,
        )
        # Indices of numeric variables on which to apply gradients
        mask = th.tensor(
            [i in problem.variables for i in input_data_shape.feature_names],
            device=self.device,
        )
        grad_indices = th.nonzero(mask, as_tuple=False).squeeze()
        input_vars_subvector = input_data.features[:, grad_indices].clone().detach()
        input_vars_subvector.requires_grad_(True)

        optimizer = optim.Adam([input_vars_subvector], lr=self.lr)
        i = 0
        while i < self.max_iter:
            input_data.features = input_data.features.clone().detach()
            input_data.features[:, grad_indices] = input_vars_subvector
            try:
                min_loss_id, min_loss = self._gradient_descent(
                    problem, input_data, optimizer=optimizer
                )
            except UncompliantSolutionError:
                pass
            else:
                if min_loss < best_loss:
                    # Compute objective, constraints and corresponding losses
                    # after reverting to raw variables
                    raw_vars = self._compute_raw_variables_from_processed_input(
                        problem=problem,
                        feature_input=input_data.features.cpu()[min_loss_id]
                        .detach()
                        .clone()
                        .reshape(1, -1),
                        make_tabular_container=make_tabular_container,
                    )
                    obj_value = problem.apply_function(problem.objective, raw_vars)
                    is_within_objective_bounds = self.within_objective_bounds(
                        obj_value.item(), problem.objective
                    )
                    loss = self.objective_loss(obj_value, problem.objective)
                    is_within_constraints = True
                    if problem.constraints:
                        constraint_loss = self.constraints_loss(
                            [
                                problem.apply_function(constraint, raw_vars)
                                for constraint in problem.constraints
                            ],
                            problem.constraints,
                        )
                        loss += constraint_loss
                        is_within_constraints = bool((constraint_loss == 0).item())
                    if (
                        is_within_objective_bounds
                        and is_within_constraints
                        and loss.item() < best_loss
                    ):
                        best_loss = loss.item()
                        best_obj = obj_value.cpu().item()
                        best_vars = raw_vars
                        best_iter = i

            # Update input_vars_subvector with constrained values
            input_vars_subvector.data = th.clip(
                input_vars_subvector.data,
                # use .data to avoid gradient tracking during update
                lower_input.features[0, grad_indices],
                upper_input.features[0, grad_indices],
            )
            if i > best_iter + self.patience:
                break
            i += 1

        if best_obj is not None and best_vars is not None:
            self._log_success(problem, i, best_obj, best_iter, best_vars)
            return best_obj, best_vars, best_loss
        else:
            self._log_failure(problem, i)
            raise NoSolutionError

    def _compute_raw_variables_from_unprocessed_input(
        self, problem: co.SOProblem, input_variable_values: Dict[str, th.Tensor]
    ) -> Dict[str, Any]:
        raw_vars: Dict[str, float] = {
            name: input_variable_values[name]
            .detach()
            .cpu()
            .numpy()
            .squeeze()
            .tolist()  # turn np.ndarray to float
            for name in problem.variables
        }
        raw_vars = {
            name: (
                int(round(val))
                if isinstance(problem.variables[name], co.IntegerVariable)
                else val
            )
            for name, val in raw_vars.items()
        }
        return raw_vars

    def _compute_raw_variables_from_processed_input(
        self,
        problem: co.SOProblem,
        feature_input: th.Tensor,
        make_tabular_container: Optional[Callable[[th.Tensor], TabularContainer]],
    ) -> Dict[str, Any]:
        raw_variable_values = {}
        if not problem.data_processor or make_tabular_container is None:
            raise Exception("Data processor or make_tabular_container is not defined!")
        feature_container = make_tabular_container(feature_input)
        best_raw_df = problem.data_processor.inverse_transform(
            feature_container, "tabular_features"
        )
        for name, variable in problem.variables.items():
            val = best_raw_df[[name]].values.squeeze().tolist()

            raw_variable_values[name] = (
                int(round(val)) if isinstance(variable, co.IntegerVariable) else val
            )
        return raw_variable_values

    def _single_start_opt(
        self,
        problem: co.SOProblem,
        seed: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float], float]:
        """Perform a single start optimization.
        Categorical variables are fixed to the values in input_parameters.
        (a grid search of categorical variables is performed in solve)
        This is where gradient descent is performed.

        Parameters
        ----------
        numeric_variables : Dict[str, co.NumericVariable]
            Numeric variables on which to apply gradients
        objective : co.Objective
            Objective to be optimized
        constraints : Sequence[co.Constraint]
            Constraints to be satisfied
        input_parameters : Optional[Dict[str, Any]], optional
            Non decision parts of the input, by default None
        seed: int, by default None
            random seed

        Returns
        -------
        Tuple[float, Dict[str, float], flat]
            - objective value
            - variables
            - best loss value

        Raises
        ------
        NoSolutionError
            No valid solution is found
        """

        if not problem.data_processor:
            return self._unprocessed_single_start_opt(problem, seed=seed)
        else:
            return self._processed_single_start_opt(problem, seed=seed)

    def solve(
        self, problem: co.SOProblem, seed: Optional[int] = None
    ) -> Tuple[float, Dict[str, float]]:
        if seed is not None:
            th.manual_seed(seed)
        if self.device:
            for constraint in problem.constraints:
                constraint.to(self.device)
            problem.objective.to(self.device)

        categorical_variables = [
            name
            for name, variable in problem.variables.items()
            if isinstance(variable, co.EnumVariable)
        ]
        numeric_variables = {
            name: variable
            for name, variable in problem.variables.items()
            if isinstance(variable, co.NumericVariable)
        }
        meshed_categorical_vars = self.get_meshed_categorical_vars(problem.variables)

        if meshed_categorical_vars is None:
            meshed_categorical_vars = np.array([0])

        best_loss_list: List[float] = []
        obj_list: List[float] = []
        vars_list: List[Dict] = []
        for i in range(self.multistart):
            for categorical_cell in meshed_categorical_vars:
                categorical_values = {
                    name: categorical_cell[ind]
                    for ind, name in enumerate(categorical_variables)
                }  # from {id: value} to {name: value}
                fixed_values = {
                    **categorical_values,
                    **(problem.input_parameters or {}),
                }
                try:
                    (
                        obj_pred,
                        best_raw_vars,
                        best_loss,
                    ) = self._single_start_opt(
                        co.SOProblem(
                            variables=numeric_variables,  # type: ignore
                            input_parameters=fixed_values,
                            objective=problem.objective,
                            constraints=problem.constraints or [],
                            data_processor=problem.data_processor,
                        ),
                        seed=seed + i if seed is not None else None,
                    )
                except NoSolutionError:
                    continue
                else:
                    best_loss_list.append(best_loss)
                    obj_list.append(obj_pred)
                    vars_list.append(best_raw_vars)
        if not obj_list:
            raise NoSolutionError("No valid solutions and variables found!")

        idx = np.argmin(best_loss_list)
        vars_cand = vars_list[idx]
        if vars_cand is not None:
            obj_cand = obj_list[idx]
            if obj_cand is None:
                raise Exception(f"Unexpected objs_list[{idx}] is None.")
        else:
            raise NoSolutionError("No valid solutions and variables found!")

        return obj_cand, vars_cand

    ##################
    ## _loss        ##
    ##################
    def constraints_loss(
        self, constraint_values: List[th.Tensor], constraints: Sequence[co.Constraint]
    ) -> th.Tensor:
        """
        compute loss of the values of each constraint function fixme: double-check

        Parameters
        ----------
        constraint_values : List[th.Tensor]
            values of each constraint function
        constraints : Sequence[co.Constraint]
            constraint functions

        Returns
        -------
        th.Tensor
            loss of the values of each constraint function

        """

        # vars: a tensor
        # get loss for constraint functions defined in the problem setting
        total_loss = th.zeros_like(
            constraint_values[0], device=self.device, dtype=self.dtype
        )
        for i, (constraint_value, constraint) in enumerate(
            zip(constraint_values, constraints)
        ):
            stress = (
                self.objective_stress
                if isinstance(constraint, co.Objective)
                else self.constraint_stress
            )
            constraint_violation = th.zeros_like(
                constraint_values[0], device=self.device, dtype=self.dtype
            )
            if constraint.upper is not None and constraint.lower is not None:
                if constraint.upper == constraint.lower:
                    constraint_violation = th.abs(constraint_value - constraint.upper)
                else:
                    normed_constraint = (constraint_value - constraint.lower) / (
                        constraint.upper - constraint.lower
                    )
                    constraint_violation = th.where(
                        (normed_constraint < 0) | (normed_constraint > 1),
                        (normed_constraint - 0.5),
                        0,
                    )
            elif constraint.lower is not None:
                constraint_violation = th.relu(constraint.lower - constraint_value)
            elif constraint.upper is not None:
                constraint_violation = th.relu(constraint_value - constraint.upper)
            total_loss += (
                constraint_violation**2 + stress * (constraint_violation > 0).float()
            )

        return total_loss

    def objective_loss(
        self, objective_value: th.Tensor, objective: co.Objective
    ) -> th.Tensor:
        """Compute the objective loss for a given objective value:
        - if no bounds are specified, use the squared objective value
        - if both bounds are specified, use the squared normalized
        objective value if it is within the bounds, otherwise
        add a stress term to a squared distance to middle of the bounds

        Parameters
        ----------
        objective_value : th.Tensor
            Tensor of objective values
        objective : co.Objective
            Objective function

        Returns
        -------
        th.Tensor
            Tensor of objective losses

        Raises
        ------
        NotImplementedError
            If only one bound is specified for the objective

        """

        if objective.upper is None and objective.lower is None:
            loss = (
                th.sign(objective_value) * (objective_value**2) * objective.direction
            )
        elif objective.upper is not None and objective.lower is not None:
            norm_cst_obj_pred = (objective_value - objective.lower) / (
                objective.upper - objective.lower
            )  # scaled
            loss = th.where(
                (norm_cst_obj_pred < 0) | (norm_cst_obj_pred > 1),
                (norm_cst_obj_pred - 0.5) ** 2 + self.objective_stress,
                norm_cst_obj_pred * objective.direction,
            )
        else:
            raise NotImplementedError("Objective with only one bound is not supported")
        return loss

    def compute_loss(self, problem: co.SOProblem, input_data: Any) -> th.Tensor:
        obj_output = problem.objective.function(input_data)
        objective_loss = self.objective_loss(obj_output, problem.objective)
        constraint_loss = th.zeros_like(objective_loss, device=self.device)

        if problem.constraints:
            const_outputs = [
                constraint.function(input_data) for constraint in problem.constraints
            ]
            constraint_loss = self.constraints_loss(const_outputs, problem.constraints)

        loss = objective_loss + constraint_loss
        return loss

    ##################
    ## _get (vars)  ##
    ##################

    def get_meshed_categorical_vars(
        self, variables: Dict[str, co.Variable]
    ) -> Optional[np.ndarray]:
        """
        Get combinations of all categorical (binary, enum) variables

        Parameters
        ----------
        variables : Dict[str, co.Variable]
            Variables to be optimized

        Returns
        -------
        Optional[np.ndarray]
            Combinations of all categorical variables
            of shape (n_samples, n_vars)
        """
        cv_value_list = [
            variable.values
            for variable in variables.values()
            if isinstance(variable, co.EnumVariable)
        ]
        if not cv_value_list:
            return None
        meshed_cv_value_list = [x_.reshape(-1, 1) for x_ in np.meshgrid(*cv_value_list)]
        meshed_cv_value = np.concatenate(meshed_cv_value_list, axis=1)
        return meshed_cv_value

    ##################
    ## _check       ##
    ##################

    # check violations of objective value var_ranges
    # reuse code in UDAO
    @staticmethod
    def within_objective_bounds(obj_value: float, objective: co.Objective) -> bool:
        """
        check whether violating the objective value var_ranges
        :param pred_dict: dict, keys are objective names,
        values are objective values
        :param obj_bounds: dict, keys are objective names,
        values are lower and upper var_ranges of each objective value
        :return: True or False
        """
        within_bounds = True
        if objective.upper is not None:
            within_bounds = obj_value <= objective.upper
        if objective.lower is not None:
            within_bounds = within_bounds and obj_value >= objective.lower
        return within_bounds
