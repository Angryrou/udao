import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .evolutionary import EVO


class GenericMOO:
    def __init__(self) -> None:
        pass

    def problem_setup(
        self,
        obj_names: list,
        obj_funcs: list,
        opt_types: list,
        const_funcs: list,
        const_types: list,
        var_types: list,
        var_ranges: np.ndarray,
        obj_types: Optional[List] = None,
        wl_list: Optional[List[str]] = None,
        wl_ranges: Optional[Dict[str, Any]] = None,
        vars_constraints: Optional[Dict] = None,
        accurate: Optional[bool] = None,
        std_func: Optional[Callable] = None,
    ) -> None:
        """
        setup common input paramters for MOO problems
        :param obj_names: list, objective names
        :param obj_funcs: list, objective functions
        :param opt_types: list, objectives to minimize or maximize
        :param const_funcs: list, constraint functions
        :param const_types: list, constraint types
            ("<=" "==" or ">=", e.g. g1(x1, x2, ...) - c <= 0)
        :param var_types: list, variable types
            (float, integer, binary, enum)
        :param var_ranges: ndarray(n_vars, ), lower and
            upper var_ranges of variables(non-ENUM), and values of ENUM variables
        :param wl_list: list, each element is a string
            to indicate workload id (fixme)
        :param wl_ranges: dict, each key is the workload id (fixme)
        :param vars_constraints: dict, keys are "conf_min" and
            "conf_max" to indicate the variable range (only used in MOGD)
        :param accurate: bool, to indicate whether the predictive
            model is accurate (True) (used in MOGD)
        :param std_func: function, used in in-accurate predictive models
        :return:
        """
        self.obj_names = obj_names
        self.obj_funcs = obj_funcs
        self.opt_types = opt_types
        self.const_funcs = const_funcs
        self.const_types = const_types
        self.var_types = var_types
        self.var_ranges = var_ranges

        # used in MOGD
        self.obj_types = obj_types
        # self.wl_list = wl_list
        self.wl_ranges = wl_ranges
        self.vars_constraints = vars_constraints
        self.accurate = accurate
        self.std_func = std_func

    def _load_job_ids(self, file_path: str) -> List[Optional[str]]:
        job_ids: List[Optional[str]] = []
        loaded_job_ids = np.loadtxt(file_path, dtype="str", delimiter=",").tolist()
        if loaded_job_ids == "None":
            job_ids = [None]
        elif isinstance(loaded_job_ids, str):
            job_ids = [loaded_job_ids]
        elif isinstance(loaded_job_ids, list):
            if "" in job_ids:
                raise Exception(f"job ids {job_ids} contain empty string!")
            else:
                pass
        else:
            raise Exception(f"job ids {job_ids} are not well defined!")
        return job_ids

    def solve(
        self, moo_algo: str, solver: str, add_params: list
    ) -> Tuple[List, List, List, List]:
        """
        solve MOO problems internally by different MOO algorithms
        :param moo_algo: str, the name of moo algorithm
        :param solver: str, the name of solver
        :param add_params: list, the parameters required by
            the specified MOO algorithm and solver
        :return:
            po_objs_list: list, each element is solutions
                (ndarray(n_solutions, n_objs)) for one job
            po_vars_list: list, each element is solutions
                (ndarray(n_solutions, n_vars)) for one job,
                corresponding variables of MOO solutions
            job_Ids: list, workload ids, each element is a string or None.
            time_cost_list: list, each element is the
            time cost of MOO solving for one job.
        """
        po_objs_list: List[Optional[np.ndarray]] = []
        po_vars_list: List[Optional[np.ndarray]] = []
        time_cost_list: List[float] = []

        if moo_algo == "evolutionary":
            file_path = add_params[0]
            job_ids = self._load_job_ids(file_path)
            inner_algo = add_params[1]
            pop_size = add_params[2]
            # the number of function evaluations
            nfe = add_params[3]
            flag = add_params[4]
            seed = add_params[5]
            evo = EVO(
                inner_algo,
                self.obj_funcs,
                self.opt_types,
                self.const_funcs,
                self.const_types,
                pop_size,
                nfe,
                fix_randomness_flag=flag,
                seed=seed,
            )

            for wl_id in job_ids:
                # fixme: to be generalized further
                if self.wl_ranges is not None and wl_id is not None:
                    vars_max, vars_min = self.wl_ranges[wl_id]
                    vars_ranges = np.vstack((vars_min, vars_max)).T
                    # find indices of non_ENUM vars
                    non_enum_inds = [
                        i
                        for i, var_type in enumerate(self.var_types)
                        if var_type != "category"
                    ]
                    vars_ranges[non_enum_inds] = self.var_ranges[non_enum_inds]
                    self.var_ranges[non_enum_inds] = list(
                        vars_ranges[non_enum_inds].tolist()
                    )
                else:
                    pass

                start_time = time.time()
                po_objs, po_vars = evo.solve(wl_id, self.var_ranges, self.var_types)
                time_cost = time.time() - start_time
                po_objs_list.append(po_objs)
                po_vars_list.append(po_vars)
                time_cost_list.append(time_cost)

        else:
            raise NotImplementedError(f"{moo_algo} is not implemented yet.")

        return po_objs_list, po_vars_list, job_ids, time_cost_list
