import random
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from platypus import (
    HUX,
    NSGAII,
    Archive,
    BitFlip,
    GAOperator,
    Integer,
    Problem,
    Real,
    nondominated,
)

from ...utils.interfaces import VarTypes


class EVO:
    def __init__(
        self,
        inner_algo: str,
        obj_funcs: list,
        opt_type: list,
        const_funcs: list,
        const_types: list,
        pop_size: int,
        nfe: int,
        fix_randomness_flag: bool,
        seed: int,
    ) -> None:
        """
        :param inner_algo: str, the name of Multi-Objective
        Evolutionary Algorithm used. By default, it is NSGA-II
        :param obj_funcs: list, objective functions
        :param opt_type: list, objectives to minimize or maximize
        :param const_funcs: list, constraint functions
        :param const_types: list, constraint types ("<=", "==" or ">=",
        e.g. g1(x1, x2, ...) - c <= 0)
        :param pop_size: int, population size
        :param nfe: int, the number of function evaluations
        :param fix_randomness_flag: bool, to indicate whether
        to fix randomness (if so, True)
        :param seed: int, the random seed to fix randomness
        """
        super().__init__()
        self.n_objs = len(obj_funcs)
        self.n_consts = len(const_funcs)
        self.obj_funcs = obj_funcs
        self.const_funcs = const_funcs
        self.const_types = const_types
        self.opt_type = opt_type
        self.inner_algo = inner_algo

        self.pop_size = pop_size
        self.nfe = nfe
        self.fix_randomness_flag = fix_randomness_flag
        if inner_algo == "NSGA-II":
            self.moo = NSGAII
        else:
            raise Exception(f"Algorithm {inner_algo} is not supported!")
        self.seed = seed

    def solve(
        self, wl_id: Optional[str], var_ranges: np.ndarray, var_types: list
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:  # type: ignore[override]
        """
        solve MOO with NSGA-II algorithm
        :param wl_id: str, workload id, e.g. '1-7'
        :param var_ranges: ndarray(n_vars,),
            lower and upper var_ranges of variables(non-ENUM),
            and values of ENUM variables
        :param var_types: list, variable types (float, integer, binary, enum)
        :return:
                po_objs: ndarray(n_solutions, n_objs),
                    Pareto solutions
                po_vars: ndarray(n_solutions, n_vars),
                    corresponding variable values of Pareto solutions
        """

        global_var_range = var_ranges
        job = wl_id

        # class to add all solutions during evolutionary iterations
        class LoggingArchive(Archive):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, *kwargs)
                self.log: Any = []

            def add(self, solution: Any) -> None:
                super().add(solution)
                self.log.append(solution)

        log_archive = LoggingArchive()

        # create a new problem
        n_vars = len(var_types)
        n_objs = self.n_objs
        n_consts = self.n_consts
        self.problem = Problem(n_vars, n_objs, n_consts)

        # set up variable types and constraint types
        flag_mixed_var_type, enum_inds = self.set_var_types(var_types, global_var_range)
        self.set_const_types()
        # pass the functions of objectives and constraints
        self.problem.function = self.get_problem_def(global_var_range, job, enum_inds)

        # fix randomness
        if self.fix_randomness_flag:
            random.seed(self.seed)

        # required by the Platypus, mixed types must define variator.
        if flag_mixed_var_type:
            algorithm = self.moo(
                self.problem,
                population_size=self.pop_size,
                variator=GAOperator(HUX(), BitFlip()),
                archive=log_archive,
            )
        else:
            algorithm = self.moo(
                self.problem, population_size=self.pop_size, archive=log_archive
            )
        algorithm.run(self.nfe)

        # find feasible solutions
        feasible_solutions_algo = [s for s in algorithm.result if s.feasible]
        if len(feasible_solutions_algo) > 0:
            # if the algorithm returns feasible solutions,
            # keep them to further find non-dominated solutions
            feasible_solutions = feasible_solutions_algo
        else:
            # if no feasible solutions are returned by the algorithm,
            # it tries to find feasible solutions among all iterations from log
            feasible_solutions_log = [s for s in log_archive.log if s.feasible]
            feasible_solutions = feasible_solutions_log

        # find non-dominated solutions
        if feasible_solutions == []:
            print(f"Evo({self.inner_algo}) cannot find feasible solutions!")
            po_objs_list, po_vars_list = None, None
            return po_objs_list, po_vars_list
        else:
            # find non-dominated solutions
            non_dominated = nondominated(feasible_solutions)
            non_dominated_objs = [
                solution.objectives._data for solution in non_dominated
            ]
            # filter duplicates
            uniq_non_dominated_objs, uniq_non_dominated_index = np.unique(
                np.array(non_dominated_objs), axis=0, return_index=True
            )
            uniq_non_dominated = np.array(non_dominated)[
                uniq_non_dominated_index
            ].tolist()
            print(
                "the number of non-dominated solutions"
                f" is {len(uniq_non_dominated_objs)}"
            )

            po_objs_list, po_vars_list = [], []
            for solution in uniq_non_dominated:
                # Within the internal Platypus library,
                # the INTEGER variable is encoded with binary numbers.
                # Here it uses decode to return back the INTEGER value
                po_vars = [
                    x.decode(y)
                    for [x, y] in zip(
                        self.problem.types, solution.variables  # type: ignore
                    )
                ]
                for i in enum_inds:
                    ind = int(po_vars[i])
                    decoded_enum = global_var_range[i][ind]
                    po_vars[i] = decoded_enum
                po_objs_list.append(solution.objectives._data)
                po_vars_list.append(po_vars)

            return np.array(po_objs_list), np.array(po_vars_list)

    def set_var_types(
        self,
        var_types: List[VarTypes],
        global_var_range: np.ndarray,
    ) -> Tuple[bool, list]:
        """
        :param var_types: list, variable types (float, integer, binary, enum)
        :return:
                flag_mixed_var_type: bool,
                to indicate whether the variable types are mixed
                (True) or all the same (False)
        """
        n_vars = len(var_types)
        # find list of indices for different variable types
        float_inds = [i for i, x in enumerate(var_types) if x == VarTypes.FLOAT]
        int_inds = [i for i, x in enumerate(var_types) if x == VarTypes.INT]
        binary_inds = [i for i, x in enumerate(var_types) if x == VarTypes.BOOL]
        # make it global as it will be used to decode
        # the solutions in the method self.solve

        enum_inds = [i for i, x in enumerate(var_types) if x == VarTypes.CATEGORY]

        # set variable types for the problem
        if len(float_inds) > 0:
            for i in float_inds:
                self.problem.types[i] = Real(
                    global_var_range[i][0], global_var_range[i][1]
                )
        if len(int_inds) > 0:
            for i in int_inds:
                self.problem.types[i] = Integer(
                    global_var_range[i][0], global_var_range[i][1]
                )
        if len(binary_inds) > 0:
            for i in binary_inds:
                self.problem.types[i] = Integer(0, 1)
        if len(enum_inds) > 0:
            # Platypus does not support the categorical variable type
            # Here the ENUM variable type is transformed by using INTEGER type to:
            # 1) indicate the indices of categorical values
            # 2) will be decoded back to the categorical values
            # when calculating values of objectives and constraints,
            #   and return final variable values of Pareto solutions
            for i in enum_inds:
                self.problem.types[i] = Integer(0, len(global_var_range[i]) - 1)

        if len(float_inds) + len(int_inds) + len(binary_inds) + len(enum_inds) == 0:
            raise Exception(
                "ERROR: No feasilbe variables provided,"
                "please check the variable types setting!"
            )
        assert (
            len(float_inds) + len(int_inds) + len(binary_inds) + len(enum_inds)
            == n_vars
        )

        # check whether the variable types are mixed
        if (
            (len(float_inds) == n_vars)
            or (len(float_inds) == n_vars)
            or (len(float_inds) == n_vars)
            or (len(float_inds) == n_vars)
        ):
            flag_mixed_var_type = False
        else:
            flag_mixed_var_type = True

        return flag_mixed_var_type, enum_inds

    def set_const_types(self) -> None:
        n_consts = len(self.const_types)
        # indices of constraint types
        le_inds = [i for i, x in enumerate(self.const_types) if x == "<="]
        ge_inds = [i for i, x in enumerate(self.const_types) if x == ">="]
        eq_inds = [i for i, x in enumerate(self.const_types) if x == "=="]

        if len(le_inds) > 0:
            for i in le_inds:
                self.problem.constraints[i] = "<=0"
        if len(ge_inds) > 0:
            for i in ge_inds:
                self.problem.constraints[i] = ">=0"
        if len(eq_inds) > 0:
            for i in eq_inds:
                self.problem.constraints[i] = "==0"

        if (len(le_inds) + len(ge_inds) + len(eq_inds) == 0) & (n_consts != 0):
            raise Exception(
                "ERROR: No feasilbe constraints provided, "
                "please check the constraint types setting!"
            )
        assert len(le_inds) + len(ge_inds) + len(eq_inds) == n_consts

    def get_problem_def(
        self, global_var_range: np.ndarray, global_job: Optional[str], enum_inds: list
    ) -> Callable:
        """
        define the problem with objective and constraints,
        only support variables as input.
        :param vars:  list, variable types (float, integer, binary, enum)
        :return:
                f_list: list, values of objective functions
                g_list: list, values of constraint functions if any
        """

        def problem_def(vars: Any) -> Union[Tuple[list, list], list]:
            if global_var_range is None or enum_inds is None:
                raise Exception("Global_var_range is not provided")
            # defined_functions need the input variables to be array
            # with shape([n, n_vars]), where n can be any positive number
            vars = np.array(vars)
            vars = vars.reshape([1, vars.shape[0]])

            if len(enum_inds) > 0:
                vars_decoded = np.ones_like(vars) * np.inf
                for i in enum_inds:
                    ind = int(vars[0, i])
                    decoded_enum = global_var_range[i][ind]
                    vars_decoded[0, i] = decoded_enum
            else:
                vars_decoded = vars

            # the formats of f_list(and g_list) is required
            # as list[value1, value2, ...]
            # f_list = [obj_func(job, vars_decoded).tolist()[0]
            # for obj_func in self.obj_funcs]
            # g_list = [const_func(job, vars_decoded).tolist()[0]
            # for const_func in self.const_funcs]

            f_list = []

            for obj_func in self.obj_funcs:
                if global_job is None:
                    obj_value = obj_func(vars_decoded).tolist()
                else:
                    obj_value = obj_func(global_job, vars_decoded).tolist()
                if isinstance(obj_value, list):
                    f_list.append(obj_value[0])
                else:
                    f_list.append(obj_value)

            if len(self.const_funcs) > 0:
                g_list = []
                for const_func in self.const_funcs:
                    if global_job is None:
                        const_value = const_func(vars_decoded).tolist()
                    else:
                        const_value = const_func(global_job, vars_decoded).tolist()
                    if isinstance(const_value, list):
                        g_list.append(const_value[0])
                    else:
                        g_list.append(const_value)

                return f_list, g_list
            else:
                return f_list

        return problem_def
