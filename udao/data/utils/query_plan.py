import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple, cast

import dgl
import networkx as nx
import numpy as np
import torch as th
import torch.nn.functional as F
from networkx.algorithms import isomorphism
from scipy import sparse as sp


def format_size(size: str) -> float:
    unit_to_multiplier = {
        "B": 1,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
    }
    n, unit = size.replace(",", "").split()

    try:
        return float(n) * unit_to_multiplier[unit]
    except KeyError:
        raise Exception(f"unkown {n} in {size}")


@dataclass
class QueryPlanOperationFeatures:
    """Features of the operations in the logical plan"""

    rows_count: List[float]
    size: List[float]

    @property
    def operation_ids(self) -> List[int]:
        return list(range(len(self.rows_count)))

    @property
    def features_dict(self) -> Dict[str, List]:
        return {"rows_count": self.rows_count, "size": self.size}

    @classmethod
    def get_feature_names_and_types(cls) -> Dict[str, type]:
        """Retrieve the names and element types of features."""
        feature_info = {}
        for f in fields(cls):
            # Assuming the type is always List[SomeType]
            try:
                element_type = f.type.__args__[0]
            except AttributeError:
                raise ValueError(f"Feature {f.name} should be a List, but got {f.type}")
            feature_info[f.name] = element_type
        return feature_info


class QueryPlanStructure:
    """Generate a graph from the tree structure of the logical plan.

    Parameters
    ----------
    node_names : List[str]
        list of the names of the nodes in the graph
    incoming_ids : List[int]
        For each edge i, incoming_ids[i] is the id of the source node
    outgoing_ids : List[int]
        for each edge i, outgoing_ids[i] is the id of the destination node
    """

    def __init__(
        self,
        node_names: List[str],
        incoming_ids: List[int],
        outgoing_ids: List[int],
    ) -> None:
        self.incoming_ids = incoming_ids
        self.outgoing_ids = outgoing_ids
        self.node_id2name = {i: n for i, n in enumerate(node_names)}
        self.graph: dgl.DGLGraph = dgl.graph(
            (incoming_ids, outgoing_ids), num_nodes=len(node_names)
        )
        self._nx_graph: Optional[nx.Graph] = None

    @property
    def nx_graph(self) -> nx.Graph:
        if self._nx_graph is None:
            self._nx_graph = dgl.to_networkx(self.graph)
            nx.set_node_attributes(self._nx_graph, self.node_id2name, name="nname")
        return self._nx_graph

    def graph_match(self, plan: "QueryPlanStructure") -> bool:
        """Checks if the current plan and another one are isomorphic.

        Parameters
        ----------
        plan : QueryPlanStructure
            another plan to compare with

        Returns
        -------
        bool
            True if the plans are isomorphic, False otherwise
        """
        matcher = isomorphism.GraphMatcher(
            self.nx_graph,
            plan.nx_graph,
            node_match=lambda n1, n2: n1["nname"] == n2["nname"],
        )
        return matcher.is_isomorphic()  # type: ignore


class _LogicalOperation:
    """Describes an operation in the logical plan, with associated features.

    Parameters
    ----------
    id : int
        id of the operation in the query plan (one id per line in the plan)
    value : str
        text of the operation (line in the string representation of the plan)
    """

    def __init__(self, id: int, value: str) -> None:
        self.id: int = id
        self.value: str = value
        self._rank: Optional[float] = None

    id: int
    value: str
    _rank: Optional[float] = None

    @property
    def rank(self) -> float:
        """Compute the rank of the operation,
        i.e. the number of spaces before the operation name."""
        if self._rank is None:
            self._rank = (len(self.value) - len(self.value.lstrip())) / 3
        return self._rank

    @rank.setter
    def rank(self, rank: float) -> None:
        self._rank = rank

    @property
    def link(self) -> str:
        """Compute the link between the current operation and the previous one,
        i.e. the character before the operation name."""
        return self.value.lstrip()[0]

    @property
    def name(self) -> str:
        """Compute the name of the operation: the first word after the link,
        or with additional term after the space in the case of a Relation operation"""
        _, b, c = re.split(r"([a-z])", self.value, 1, flags=re.I)
        name = (b + c).split(" ")[0]
        if name == "Relation":
            name = (b + c).split("[")[0]
        return name

    @property
    def size(self) -> float:
        """Extract the size of the operation, in bytes"""
        return format_size(self.value.split("sizeInBytes=")[1].split("B")[0] + "B")

    def get_rows_count(
        self, id_predecessors: List[int], rows_count: List[float]
    ) -> float:
        """Extract the number of rows of the operation,
        or infer it from its predecessors if not available"""
        if "rowCount=" in self.value:
            return float(self.value.split("rowCount=")[1].split(")")[0])
        else:
            if len(id_predecessors) == 1:
                pid = id_predecessors[0]
                if rows_count[pid] is None:
                    raise ValueError(
                        "The number of rows of the parent should be a float."
                        f"Got None for {pid}"
                    )
                return rows_count[pid]
            elif len(id_predecessors) == 2:
                pid1, pid2 = id_predecessors
                if rows_count[pid1] is None or rows_count[pid2] is None:
                    raise ValueError(
                        "The number of rows of the parent should be a float, but "
                        f"got None for {pid1}: {rows_count[pid1]}"
                        f"or {pid2}: {rows_count[pid2]}"
                    )
                if rows_count[pid1] * rows_count[pid2] != 0:
                    raise ValueError(
                        "The product of number of rows of parents should be 0, but "
                        f"got {rows_count[pid1] * rows_count[pid2]}"
                    )
                return 0
            else:
                raise NotImplementedError("More than 2 predecessors")

    def get_unindented(self) -> "_LogicalOperation":
        """Return the operation without one level of indentation removed"""
        return _LogicalOperation(self.id, self.value.lstrip()[3:])


def _get_tree_structure(
    operations: List[_LogicalOperation],
) -> Tuple[List[int], List[int]]:
    """Define the tree structure of the logical plan, as a tuple of two lists,
    where U[i], V[i] are the incoming and outgoing nodes of edge i.

    Parameters
    ----------
    steps : List[LogicalStep]
        List of operation in the logical plan

    Returns
    -------
    Tuple[List[int], List[int]]
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    incoming_ids: List[int] = []
    outgoing_ids: List[int] = []
    pre_id = operations[0].id
    operations[0].rank = -1
    threads: Dict[float, List[_LogicalOperation]] = defaultdict(list)
    for i, op in enumerate(operations):
        if i == 0:
            continue
        if op.link == ":":
            unindented_step = op.get_unindented()
            threads[op.rank].append(unindented_step)

        elif op.link == "+":
            incoming_ids.append(op.id)
            outgoing_ids.append(pre_id)
            pre_id = op.id
        else:
            raise ValueError(
                f"Unexpected link {op.link} in {op.value}. Expected : or +"
            )

    for rank, rank_operations in threads.items():
        incoming_ids.append(rank_operations[0].id)
        outgoing_id_candidates = [
            op.id for op in operations if op.rank == rank - 1 and op.link != ":"
        ]
        if len(outgoing_id_candidates) != 1:
            warnings.warn(
                f"More than one candidate outgoing id for rank {rank}."
                "\n Taking the first candidate."
            )
        outgoing_ids.append(outgoing_id_candidates[0])
        sub_from_ids, sub_to_ids = _get_tree_structure(rank_operations)
        incoming_ids += sub_from_ids
        outgoing_ids += sub_to_ids
    return incoming_ids, outgoing_ids


def compute_meta_features(
    structure: QueryPlanStructure, features: QueryPlanOperationFeatures
) -> Dict[str, float]:
    """Compute the meta features of the operations in the query plan."""
    graph = structure.graph
    input_meta = {}
    for op_feature, feature_values in features.features_dict.items():
        graph.ndata[op_feature] = th.tensor(feature_values)
    input_nodes_index = th.where(graph.in_degrees() == 0)[0]
    for op_feature in features.features_dict.keys():
        input_meta[f"meta_{op_feature}"] = (
            graph.ndata[op_feature][input_nodes_index].sum(0).item()
        )
        del graph.ndata[op_feature]
    return input_meta


def get_laplacian_positional_encoding(
    graph: dgl.DGLGraph, pos_enc_dim: Optional[int] = None
) -> th.Tensor:
    """
    Graph positional encoding v/ Laplacian eigenvectors

    Parameters
    ----------
    graph : dgl.DGLGraph
        graph to encode
    pos_enc_dim : Optional[int], optional
        number of eigenvectors to use,
        by default set to the number of nodes in the graph - 2
    """
    if pos_enc_dim is None:
        pos_enc_dim = graph.number_of_nodes() - 2
    # Laplacian
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = graph.adj_external(scipy_fmt="csr").T  # adjacency matrix

    in_degrees = cast(th.Tensor, graph.in_degrees()).numpy().clip(1)
    N = sp.diags(in_degrees**-0.5, dtype=float, format="csr")
    L = sp.eye(graph.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    eigen_values, eigen_vectors = np.linalg.eig(L.toarray())
    idx = eigen_values.argsort()  # increasing order
    eigen_values, eigen_vectors = eigen_values[idx], np.real(eigen_vectors[:, idx])
    return th.from_numpy(eigen_vectors[:, 1 : pos_enc_dim + 1]).float()


def add_positional_encoding(graph: dgl.DGLGraph, size: int) -> dgl.DGLGraph:
    """Add positional encoding to the graph.

    Parameters
    ----------
    graph : dgl.DGLGraph
        graph to encode
    size : int
        size of the positional encoding
        (truncated if the graph has less nodes or padded with 0 otherwise)

    Returns
    -------
    dgl.DGLGraph
        graph with positional encoding as "pos_enc" node feature
    """
    bidirectional = cast(dgl.DGLGraph, dgl.to_bidirected(graph))
    graph.ndata["pos_enc"] = resize_tensor(
        get_laplacian_positional_encoding(bidirectional, graph.num_nodes() - 2), size
    )
    return graph


def resize_tensor(tensor: th.Tensor, new_size: int) -> th.Tensor:
    """Pad or truncate the last dimension of a tensor to a given size."""
    current_size = tensor.shape[-1]
    if current_size < new_size:
        tensor = F.pad(
            tensor,
            pad=(0, new_size - current_size),
            mode="constant",
            value=0,
        )
    else:
        # truncate last dimension
        tensor = tensor[..., :new_size]

    return tensor


def random_flip_positional_encoding(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    """Add same random sign flip to all nodes positional encodings of the graph."""
    pos_enc = cast(th.Tensor, graph.ndata["pos_enc"])
    sign_flip = th.rand(pos_enc.size(1))
    sign_flip[sign_flip >= 0.5] = 1.0
    sign_flip[sign_flip < 0.5] = -1.0
    pos_enc *= sign_flip.unsqueeze(0)
    graph.ndata["pos_enc"] = pos_enc
    return graph


def extract_query_plan_features(
    logical_plan: str,
) -> Tuple[QueryPlanStructure, QueryPlanOperationFeatures, Dict[str, float]]:
    """Extract:
    - features of the operations in the logical plan
    - meta features aggregated from operations of the logical plan
    - the tree structure of the logical plan


    Parameters
    ----------
    logical_plan : str
        string representation of the logical plan

    Returns
    -------
    Tuple[QueryPlanStructure, QueryPlanOperationFeatures, Dict[str, float]]
        Query plan structure (dgl graph with node names)
        Query plan features (sizes and rows_counts of the operations)
        Meta features (aggregated features of the operations)
    """
    operations = [
        _LogicalOperation(id=i, value=step)
        for i, step in enumerate(logical_plan.splitlines())
    ]
    node_names: List[str] = []
    sizes: List[float] = []
    rows_counts: List[float] = []

    incoming_ids, outgoing_ids = _get_tree_structure(operations)
    predecessors: Dict[int, List[int]] = defaultdict(list)
    for f, t in zip(incoming_ids, outgoing_ids):
        predecessors[t].append(f)

    for step in reversed(operations):
        node_names.insert(0, step.name)
        sizes.insert(0, step.size)
        rows_counts.insert(0, step.get_rows_count(predecessors[step.id], rows_counts))

    op_features = QueryPlanOperationFeatures(rows_count=rows_counts, size=sizes)
    structure = QueryPlanStructure(
        node_names=node_names, incoming_ids=incoming_ids, outgoing_ids=outgoing_ids
    )
    meta_features = compute_meta_features(structure, op_features)
    return structure, op_features, meta_features
