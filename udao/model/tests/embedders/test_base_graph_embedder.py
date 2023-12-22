import pytest
import torch as th

from ...embedders.base_graph_embedder import BaseGraphEmbedder
from .conftest import generate_dgl_graph


def test_base_embedder_initialization() -> None:
    params = BaseGraphEmbedder.Params(
        input_size=5,
        output_size=10,
        op_groups=["type", "cbo"],
        type_embedding_dim=5,
        embedding_normalizer="BN",
        n_op_types=3,
    )

    embedder = BaseGraphEmbedder(params)

    assert embedder.input_size == 10
    assert embedder.embedding_size == 10
    assert embedder.op_type
    assert embedder.op_cbo
    assert not embedder.op_enc


def test_base_embedder_invalid_normalizer() -> None:
    params = BaseGraphEmbedder.Params(
        input_size=5,
        output_size=10,
        op_groups=["type", "cbo"],
        type_embedding_dim=5,
        embedding_normalizer="UNKNOWN",  # type: ignore
        n_op_types=3,
    )

    with pytest.raises(ValueError):
        BaseGraphEmbedder(params)


def test_base_embedder_concatenate_op_features() -> None:
    params = BaseGraphEmbedder.Params(
        input_size=5,
        output_size=10,
        op_groups=["type", "cbo"],
        type_embedding_dim=5,
        embedding_normalizer=None,
        n_op_types=3,
    )

    embedder = BaseGraphEmbedder(params)
    g = generate_dgl_graph(
        3,
        2,
        {"op_gid": {"size": 1, "type": "int"}, "cbo": {"size": 2, "type": "float"}},
    )
    result = embedder.concatenate_op_features(g)

    assert isinstance(result, th.Tensor)
    assert result.shape == (3, 5 + 2)  # 5 for embedding and 2 for "cbo"


def test_base_embedder_normalize_embedding() -> None:
    params = BaseGraphEmbedder.Params(
        input_size=5,
        output_size=3,
        op_groups=["type"],
        type_embedding_dim=5,
        embedding_normalizer="BN",
        n_op_types=3,
    )

    embedder = BaseGraphEmbedder(params)
    tensor = th.randn(5, 3)

    normalized_tensor = embedder.normalize_embedding(tensor)

    assert isinstance(normalized_tensor, th.Tensor)
    assert normalized_tensor.shape == tensor.shape
