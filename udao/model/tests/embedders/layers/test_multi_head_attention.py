from typing import Tuple

import dgl
import pytest
import torch as th

from ....embedders.layers.multi_head_attention import (
    MultiHeadAttentionLayer,
    QFMultiHeadAttentionLayer,
    RAALMultiHeadAttentionLayer,
)
from ....utils.utils import set_deterministic_torch

FixtureType = Tuple[MultiHeadAttentionLayer, dgl.DGLGraph, th.Tensor]


@pytest.fixture
def mha_fixture() -> FixtureType:
    g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    h = th.rand((4, 5))
    mha = MultiHeadAttentionLayer(in_dim=5, out_dim=3, n_heads=2, use_bias=True)

    return mha, g, h


class TestMultiHeadAttentionLayer:
    def test_init(self) -> None:
        mha = MultiHeadAttentionLayer(in_dim=5, out_dim=3, n_heads=2, use_bias=True)
        assert mha.Q.in_features == 5
        assert mha.Q.out_features == 6
        assert mha.K.in_features == 5
        assert mha.K.out_features == 6
        assert mha.V.in_features == 5
        assert mha.V.out_features == 6

    def test_forward(self, mha_fixture: FixtureType) -> None:
        set_deterministic_torch(0)
        mha, g, h = mha_fixture
        out = mha(g, h)
        expected = th.tensor(
            [
                [
                    [0.000000e00, 0.000000e00, 0.000000e00],
                    [0.000000e00, 0.000000e00, 0.000000e00],
                ],
                [
                    [-5.366904e-01, 6.547331e-02, -7.822125e-02],
                    [1.462809e-01, -3.937286e-01, -7.183467e-01],
                ],
                [
                    [-5.425442e-01, -1.044514e-02, -1.592951e-01],
                    [-1.745791e-01, -5.798197e-01, -6.325653e-01],
                ],
                [
                    [-7.297680e-01, 9.695685e-02, -3.332692e-01],
                    [-6.688768e-03, -5.367336e-01, -7.892159e-01],
                ],
            ]
        )
        assert out.size() == (4, 2, 3)
        assert th.allclose(out[1:], expected[1:], rtol=1e-5)

    def test_compute_query_key_value(self, mha_fixture: FixtureType) -> None:
        mha, g, h = mha_fixture
        g = mha.compute_query_key_value(g, h)
        for key in ["Q_h", "K_h", "V_h"]:
            assert key in g.ndata
            assert g.ndata[key].size() == (4, 2, 3)  # type: ignore
        pass

    def test_compute_attention(self, mha_fixture: FixtureType) -> None:
        mha, g, h = mha_fixture
        graph = mha.compute_query_key_value(g, h)
        g = mha.compute_attention(graph)
        assert "score" in g.edata

    def test_attention_has_effect(self, mha_fixture: FixtureType) -> None:
        mha, g, h = mha_fixture
        out = mha(g, h)
        original_h = h.clone()
        assert not th.equal(original_h, out[:, 0, :])


class TestQFMultiHeadAttentionLayer:
    def test_compute_attention(self) -> None:
        set_deterministic_torch(0)
        g = dgl.graph(([0, 1, 2], [1, 2, 3]))
        h = th.rand((4, 5))
        g.edata["dist"] = th.tensor([1, 2, 3])
        attention_bias = th.tensor([0.1, 0.2, 0.3])
        mha = QFMultiHeadAttentionLayer(
            in_dim=5,
            out_dim=3,
            n_heads=2,
            use_bias=True,
            attention_bias=attention_bias,
        )
        g = mha.compute_query_key_value(g, h)
        g = mha.compute_attention(g)
        expected_score = th.tensor(
            [
                [[9.535783e-01], [1.067863e00]],
                [[1.099911e00], [1.133566e00]],
                [[1.203815e00], [1.290228e00]],
            ]
        )

        assert "score" in g.edata
        assert th.allclose(g.edata["score"], expected_score, rtol=1e-5)  # type: ignore


class TestRAALMultiHeadAttentionLayer:
    def test_compute_attention(self) -> None:
        set_deterministic_torch(0)
        g = dgl.graph(([0, 1, 2], [1, 2, 3]))
        g.ndata["sid"] = th.tensor([0, 0, 0, 0])
        non_siblings_map = {0: {0: [2, 3], 1: [0, 3], 2: [0, 1]}}
        h = th.rand((4, 5))
        mha = RAALMultiHeadAttentionLayer(
            in_dim=5,
            out_dim=3,
            n_heads=2,
            use_bias=True,
            non_siblings_map=non_siblings_map,
        )
        g = mha.compute_query_key_value(g, h)
        g = mha.compute_attention(g)
        expected_score = th.tensor(
            [
                [[3.676294e-01], [8.327479e-01]],
                [[2.522929e-01], [3.981344e00]],
                [[6.267625e-01], [3.711438e-01]],
            ]
        )
        assert "score" in g.edata
        assert th.allclose(g.edata["score"], expected_score, rtol=1e-5)  # type: ignore
