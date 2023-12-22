from typing import cast

import dgl
import pytest
import torch as th

from ...embedders.graph_transformer import GraphTransformer
from ...embedders.layers.multi_head_attention import AttentionLayerName
from ...utils.utils import set_deterministic_torch
from .conftest import generate_dgl_graph


def test_graph_transformer_initialization() -> None:
    for attention_layer in ["QF", "RAAL", "GTN"]:
        net_params = GraphTransformer.Params(
            input_size=3,
            output_size=6,
            op_groups=["type", "cbo"],
            type_embedding_dim=3,
            embedding_normalizer="BN",
            n_op_types=4,
            pos_encoding_dim=5,
            n_layers=3,
            n_heads=2,
            hidden_dim=6,
            readout="sum",
            attention_layer_name=cast(AttentionLayerName, attention_layer),
            dropout=0.1,
            max_dist=3 if attention_layer == "QF" else None,
            non_siblings_map={0: {0: [0, 1, 2], 1: [1, 2, 3]}}
            if attention_layer == "RAAL"
            else None,
        )

        model = GraphTransformer(net_params)
        assert isinstance(model, GraphTransformer)


def test_graph_transformer_initialization_raises_error() -> None:
    with pytest.raises(ValueError):
        net_params = GraphTransformer.Params(
            input_size=3,
            output_size=3,
            op_groups=["type", "cbo"],
            type_embedding_dim=3,
            embedding_normalizer="BN",
            n_op_types=4,
            pos_encoding_dim=5,
            n_layers=3,
            n_heads=2,
            hidden_dim=6,
            readout="sum",
            dropout=0.1,
        )
        GraphTransformer(net_params)


def test_graph_transformer_forward() -> None:
    set_deterministic_torch(0)
    net_params = GraphTransformer.Params(
        input_size=3,
        output_size=4,
        op_groups=["type", "cbo"],
        type_embedding_dim=5,
        embedding_normalizer="BN",
        n_op_types=5,
        pos_encoding_dim=5,
        n_layers=3,
        n_heads=2,
        hidden_dim=4,
        readout="sum",
        attention_layer_name="GTN",
        dropout=0.1,
    )

    transformer = GraphTransformer(net_params)
    features = {
        "op_gid": {"type": "int", "size": 1},
        "cbo": {"type": "float", "size": 3},
    }

    g1 = generate_dgl_graph(10, 9, features)
    g2 = generate_dgl_graph(
        5,
        4,
        features,
    )

    g_batch = dgl.batch([g1, g2])
    h_lap_pos_enc = th.randn(10 + 5, 5)
    g_batch.ndata["pos_enc"] = h_lap_pos_enc
    transformer.eval()
    output = transformer.forward(g_batch)
    expected = th.tensor(
        [
            [4.184283, -2.093954, 11.841289, 16.731495],
            [1.523479, -0.436974, 7.724406, 4.564045],
        ]
    )
    assert output.shape == (2, net_params.output_size)
    assert th.allclose(output, expected, atol=1e-5)
