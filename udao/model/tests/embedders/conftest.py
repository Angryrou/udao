import dgl
import pytest
import torch as th


def generate_dgl_graph(num_nodes: int, num_edges: int, features: dict) -> dgl.DGLGraph:
    u = th.tensor(range(num_edges))
    v = th.tensor(range(1, num_edges + 1))
    g = dgl.graph((u, v))
    for feat_name, props in features.items():
        if props["type"] == "float":
            g.ndata[feat_name] = th.randn(num_nodes, props["size"])
        elif props["type"] == "int":
            g.ndata[feat_name] = th.randint(0, 3, (num_nodes, props["size"]))
        if props["size"] == 1:
            g.ndata[feat_name] = g.ndata[feat_name].squeeze()  # type: ignore
    return g


@pytest.fixture
def generate_dgl_graph_fixture(request: pytest.FixtureRequest) -> dgl.DGLGraph:
    """
    Generate a DGL graph.

    Parameters
    - num_nodes: number of nodes
    - num_edges: number of edges
    - features: a dictionary with node/edge feature names as keys
                and feature sizes as values
    """
    num_nodes = request.param.get("num_nodes", 2)
    num_edges = request.param.get("num_edges", 2)
    features = request.param.get("features", {})
    g = generate_dgl_graph(num_nodes, num_edges, features)

    return g
