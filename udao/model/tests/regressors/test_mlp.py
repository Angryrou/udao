import pytest
import torch as th

from ...regressors.mlp import MLP
from ...utils.utils import set_deterministic_torch


@pytest.mark.parametrize(
    "embed_dim, feat_dim, expected_output",
    [
        (
            10,
            10,
            th.tensor(
                [
                    [1.175319e00, 2.838397e00, 1.457014e00, 1.331543e00, 6.580783e-01],
                    [
                        1.730856e00,
                        6.496964e-01,
                        3.652042e-01,
                        7.815728e-01,
                        8.088993e-01,
                    ],
                ]
            ),
        ),
        (
            5,
            5,
            th.tensor(
                [
                    [1.338640e00, 2.081545e00, 1.356574e00, 5.648745e-01, 5.369927e-01],
                    [1.742966e00, 8.478099e-01, 1.723262e00, 1.659415e00, 8.338411e-01],
                ]
            ),
        ),
    ],
)
def test_udao_mlp_forward_shape(
    embed_dim: int, feat_dim: int, expected_output: th.Tensor
) -> None:
    set_deterministic_torch(0)
    sample_mlp_params = MLP.Params(
        input_embedding_dim=embed_dim,
        input_features_dim=feat_dim,
        output_dim=5,
        n_layers=3,
        hidden_dim=5,
        dropout=0,
    )
    model = MLP(sample_mlp_params)
    sample_embedding = th.rand((2, embed_dim))
    sample_inst_feat = th.rand((2, feat_dim))
    output = model.forward(sample_embedding, sample_inst_feat)
    assert output.shape == (2, 5)
    assert th.allclose(output, expected_output, rtol=1e-5)
