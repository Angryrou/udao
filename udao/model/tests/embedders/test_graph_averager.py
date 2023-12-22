import dgl
import pytest
import torch as th

from ...embedders.graph_averager import GraphAverager
from ...utils.utils import set_deterministic_torch
from .conftest import generate_dgl_graph


@pytest.fixture
def params_fixture() -> GraphAverager.Params:
    return GraphAverager.Params(
        input_size=2,
        output_size=4,
        op_groups=["type", "cbo"],
        type_embedding_dim=5,
        embedding_normalizer=None,
        n_op_types=3,
    )


class TestAverager:
    def test_forward_shape(self, params_fixture: GraphAverager.Params) -> None:
        set_deterministic_torch(0)
        averager = GraphAverager(params_fixture)
        features_dict = {
            "op_gid": {"size": 1, "type": "int"},
            "cbo": {"size": 2, "type": "float"},
        }
        g1 = generate_dgl_graph(3, 2, features_dict)
        g2 = generate_dgl_graph(5, 4, features_dict)
        g_batch = dgl.batch([g1, g2])
        embedding = averager.forward(g_batch)

        expected = th.tensor(
            [
                [0.399936, 0.393089, 0.000000, 0.469329],
                [0.357530, 0.572405, 0.000000, 0.614257],
            ]
        )
        assert th.isclose(embedding, expected, rtol=1e-5).all()
        assert embedding.shape == (2, 4)
