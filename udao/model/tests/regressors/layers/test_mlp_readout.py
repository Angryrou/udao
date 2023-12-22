from typing import List, Optional

import pytest
import torch as th

from ....regressors.layers.mlp_readout import MLPReadout
from ....utils.utils import set_deterministic_torch


@pytest.mark.parametrize("agg_dims", [None, [10, 15], [12]])
def test_initialize_with_aggregation_layers(agg_dims: Optional[List[int]]) -> None:
    model = MLPReadout(10, 20, 5, dropout=0, n_layers=3, agg_dims=agg_dims)

    total_layers = 3 + 1  # inner layers + output layer
    if agg_dims:
        total_layers += len(agg_dims)

    assert len(model.FC_layers) == total_layers
    assert model.BN_layers is not None
    assert len(model.BN_layers) == total_layers - 1


def test_init_with_dropout() -> None:
    model = MLPReadout(10, 20, 5, dropout=0.5, n_layers=3, agg_dims=[10, 15])
    assert model.BN_layers is None


@pytest.mark.parametrize(
    "agg_dims, dropout, expected_output",
    [
        (
            None,
            0,
            th.tensor([[-9.637698e-02, -5.446782e-02], [5.620091e-02, -3.493200e-02]]),
        ),
        (
            [10, 15],
            0,
            th.tensor([[-5.963351e-01, 4.445746e-02], [1.587002e-01, 2.566562e-01]]),
        ),
        (
            [12],
            0,
            th.tensor([[-1.248996e-01, -2.419595e-02], [3.839206e-01, -1.896000e-01]]),
        ),
    ],
)
def test_mlp_readout_forward(
    agg_dims: Optional[List[int]], dropout: float, expected_output: th.Tensor
) -> None:
    set_deterministic_torch(0)
    model = MLPReadout(10, 20, 2, dropout=dropout, n_layers=3, agg_dims=agg_dims)
    input_tensor = th.rand((2, 10))
    output = model.forward(input_tensor)
    assert output.shape == (2, 2)
    assert th.allclose(output, expected_output, rtol=1e-5)
