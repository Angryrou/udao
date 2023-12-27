from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch as th
from matplotlib import pyplot as plt

from .exceptions import NoSolutionError


class Point:
    def __init__(self, objs: np.ndarray, vars: Optional[Dict] = None) -> None:
        """
        A point in the objective space.
        Variables are optional, and are not specified for imaginary points
        (e.g., utopia and nadir)

        Parameters
        ----------
        objs : np.ndarray
            Array of objective values of shape (n_objs,)
        vars :np.ndarray, optional
            Array of variable values of shape (n_vars,), by default None
        """
        self.objs = objs
        self.vars = vars
        self.n_objs = objs.shape[0]

    def __repr__(self) -> str:
        return f"Point(objs={self.objs}, vars={self.vars})"

    def __eq__(self, other: "Point") -> bool:  # type: ignore
        return bool(np.all(self.objs == other.objs) and np.all(self.vars == other.vars))


class Rectangle:
    def __init__(self, utopia: Point, nadir: Point) -> None:
        """

        Parameters
        ----------
        utopia : Points
            utopia point
        nadir : Points
            nadir point
        """

        self.upper_bounds = nadir.objs
        self.lower_bounds = utopia.objs
        self.n_objs = nadir.objs.shape[0]
        self.volume = self.cal_volume(nadir.objs, utopia.objs)
        self.neg_vol = -self.volume
        self.utopia = utopia
        self.nadir = nadir

    def __repr__(self) -> str:
        return f"Rectangle(utopia={self.utopia}, nadir={self.nadir})"

    def cal_volume(self, upper_bounds: np.ndarray, lower_bounds: np.ndarray) -> float:
        """
        Calculate the volume of the hyper_rectangle

        Parameters
        ----------
        upper_bounds : np.ndarray(
            Array of upper bounds of the hyper_rectangle, of shape (n_objs,)
        lower_bounds : np.ndarrays
            Array of lower bounds of the hyper_rectangle of shape (n_objs,)

        Returns
        -------
        float
            volume of the hyper_rectangle
        """
        volume = np.abs(np.prod(upper_bounds - lower_bounds))
        return volume

    # Override the `__lt__()` function to make `Rectangles`
    # class work with min-heap (referred from VLDB2022)
    def __lt__(self, other: "Rectangle") -> bool:
        return self.neg_vol < other.neg_vol

    def __eq__(self, other: "Rectangle") -> bool:  # type: ignore
        return bool(
            np.all(self.upper_bounds == other.upper_bounds)
            and np.all(self.lower_bounds == other.lower_bounds)
        )


# a quite efficient way to get the indexes of pareto points
# https://stackoverflow.com/a/40239615
def is_pareto_efficient(costs: np.ndarray, return_mask: bool = True) -> np.ndarray:
    ## reuse code in VLDB2022
    """
    Find the pareto-efficient points

    Parameters
    ----------
    costs : np.ndarray
        An (n_points, n_costs) array
    return_mask : bool, default=True
        True to return a mask

    Returns
    -------
    np.ndarray
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def summarize_ret(
    po_obj_list: Sequence, po_var_list: Sequence
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the pareto-optimal objectives and variables

    Parameters
    ----------
    po_obj_list: Sequence
        List of objective values
    po_var_list : _type_
        List of variable values

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        Pareto-optimal objectives and variables
    """
    ## reuse code in VLDB2022
    if len(po_obj_list) == 0:
        raise NoSolutionError("No feasible solutions found: empty po_obj_list")
    elif len(po_obj_list) == 1:
        return np.array(po_obj_list), np.array(po_var_list)
    else:
        po_objs_cand = np.array(po_obj_list)
        po_vars_cand = np.array(po_var_list)
        po_inds = is_pareto_efficient(po_objs_cand)
        po_objs = po_objs_cand[po_inds]
        po_vars = po_vars_cand[po_inds]

        return po_objs, po_vars


# generate even weights for 2d and 3D
def even_weights(stepsize: float, n_objectives: int) -> np.ndarray:
    """Generate even weights for 2d and 3D

    Parameters
    ----------
    stepsize : float
        Step size for the weights
    n_objectives : int
        Number of objectives for which to generate weights
        Only 2 and 3 are supported

    Returns
    -------
    np.ndarray
        Array of weights of shape (n_weights, n_objectives)

    Raises
    ------
    Exception
        If `n_objectives` is not 2 or 3
    """
    ws_pairs = np.array([])
    if n_objectives == 2:
        w1 = np.hstack([np.arange(0, 1, stepsize), 1])
        w2 = 1 - w1
        ws_pairs = np.array([[w1, w2] for w1, w2 in zip(w1, w2)])

    elif n_objectives == 3:
        w_steps = np.linspace(0, 1, num=int(1 / stepsize) + 1, endpoint=True)
        for i, w in enumerate(w_steps):
            # use round to avoid case of floating point limitations in Python
            # the limitation: 1- 0.9 = 0.09999999999998 rather than 0.1
            other_ws_range = round((1 - w), 10)
            w2 = np.linspace(
                0,
                other_ws_range,
                num=round(other_ws_range / stepsize + 1),
                endpoint=True,
            )
            w3 = other_ws_range - w2
            num = w2.shape[0]
            w1 = np.array([w] * num)
            ws = np.hstack(
                [w1.reshape([num, 1]), w2.reshape([num, 1]), w3.reshape([num, 1])]
            )
            if i == 0:
                ws_pairs = ws
            else:
                ws_pairs = np.vstack([ws_pairs, ws])
    else:
        raise Exception(f"{n_objectives} objectives are not supported.")

    assert all(np.round(np.sum(ws_pairs, axis=1), 10) == 1)
    return np.array(ws_pairs)


def plot_po(po: np.ndarray, n_obj: int = 2, title: str = "pf_ap") -> None:
    """Plot pareto-optimal solutions"""
    # po: ndarray (n_solutions * n_objs)
    ## for 2d
    if n_obj == 2:
        po_obj1 = po[:, 0]
        po_obj2 = po[:, 1]

        fig, ax = plt.subplots()
        ax.scatter(po_obj1, po_obj2, marker="o", color="blue")
        ax.plot(po_obj1, po_obj2, color="blue")

        ax.set_xlabel("Obj 1")
        ax.set_ylabel("Obj 2")

        ax.set_title(title)

    elif n_obj == 3:
        po_obj1 = po[:, 0]
        po_obj2 = po[:, 1]
        po_obj3 = po[:, 2]

        plt.figure()
        ax = plt.axes(projection="3d")

        # ax.plot_trisurf(po_obj1, po_obj2, po_obj3, antialiased=True)
        # ax.plot_surface(po_obj1, po_obj2, po_obj3)
        ax.scatter3D(po_obj1, po_obj2, po_obj3, color="blue")

        ax.set_xlabel("Obj 1")
        ax.set_ylabel("Obj 2")
        ax.set_zlabel("Obj 3")

    else:
        raise Exception(
            f"{n_obj} objectives are not supported in the code repository for now!"
        )

    plt.tight_layout()
    plt.show()


def get_default_device() -> th.device:
    return th.device("cuda") if th.cuda.is_available() else th.device("cpu")
