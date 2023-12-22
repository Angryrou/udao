from typing import List, Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MLPReadout(nn.Module):
    """MLP inner layers for regression.

    Parameters
    ----------
    input_dim : int
        Dimension of the input tensor
    hidden_dim : int
        Dimension of the hidden layers
    output_dim : int
        Dimension of the output tensor
    n_layers : int, optional
        Number of fully connected layers, by default 2
    dropout : float, optional
        Dropout probability, by default 0
        If no dropout is set, batch normalization is used instead.
    agg_dims : Optional[List[int]], optional
        Dimensions of the aggregation layers, by default None
        None means no aggregation layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        dropout: float = 0,
        agg_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.agg_dims = agg_dims
        self.FC_layers = self._load_FC_layers()
        self.BN_layers = self._load_BN_layers()

        self.dropout = dropout

    def _load_FC_layers(self) -> nn.ModuleList:
        """Create the list of fully connected layers."""
        list_FC_layers = [nn.Linear(self.input_dim, self.hidden_dim)]
        for _ in range(self.n_layers - 1):
            list_FC_layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim)])
        if self.agg_dims is None:
            list_FC_layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        else:
            agg_input_dim = self.hidden_dim
            for agg_output_dim in self.agg_dims:
                list_FC_layers.append(nn.Linear(agg_input_dim, agg_output_dim))
                agg_input_dim = agg_output_dim
            list_FC_layers.append(nn.Linear(agg_input_dim, self.output_dim))
        return nn.ModuleList(list_FC_layers)

    def _load_BN_layers(self) -> Optional[nn.ModuleList]:
        """
        Create the list of batch normalization layers,
        if dropout is not used.
        One for each FC layer except last one.
        One for each aggregation layer.
        """
        if self.dropout == 0:
            list_BN_layers = [nn.BatchNorm1d(self.hidden_dim)]
            list_BN_layers.extend(
                [nn.BatchNorm1d(self.hidden_dim) for _ in range(self.n_layers - 1)]
            )
            if self.agg_dims is not None:
                list_BN_layers.extend([nn.BatchNorm1d(a) for a in self.agg_dims])
            return nn.ModuleList(list_BN_layers)
        return None

    def forward(self, input: th.Tensor) -> th.Tensor:
        y = input
        for layer_id in range(len(self.FC_layers) - 1):
            y = self.FC_layers[layer_id](y)
            y = F.relu(y)
            if self.BN_layers is not None:
                y = self.BN_layers[layer_id](y)
            else:
                y = F.dropout(y, self.dropout, training=self.training)
        return self.FC_layers[-1](y)
