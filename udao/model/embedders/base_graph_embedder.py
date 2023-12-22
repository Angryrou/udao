from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence

import dgl
import torch as th
import torch.nn as nn

from ...utils.interfaces import UdaoEmbedItemShape
from ...utils.logging import logger
from .base_embedder import BaseEmbedder
from .layers.iso_bn import IsoBN

NormalizerType = Literal["BN", "LN", "IsoBN"]


class BaseGraphEmbedder(BaseEmbedder, ABC):
    """Base class for Embedder networks.
    Takes care of preparing the input features for the
    embedding layer, and normalizing the output embedding.

    Parameters
    ----------
    net_params : EmbedderParams
    """

    @dataclass
    class Params(BaseEmbedder.Params):
        input_size: int  # depends on the data
        """The size of the input features, except for the type of operation.
        If type is provided, the input size is increased at init
        by the type embedding dimension.
        """
        n_op_types: Optional[int]  # depends on the data
        """The number of operation types."""
        op_groups: Sequence[str]
        """The groups of operation features to be included in the embedding."""
        type_embedding_dim: Optional[int]
        """The dimension of the operation type embedding."""
        embedding_normalizer: Optional[NormalizerType]
        """Name of the normalizer to use for the output embedding."""

    @classmethod
    def from_iterator_shape(
        cls,
        iterator_shape: UdaoEmbedItemShape[Dict[str, int]],
        **kwargs: Any,
    ) -> "BaseGraphEmbedder":
        embedding_input_shapes = iterator_shape.embedding_input_shape
        op_groups = [name for name in embedding_input_shapes.keys()]
        input_size = sum(
            [embedding_input_shapes[name] for name in op_groups if name != "type"]
        )
        n_op_types = None
        if "type" in op_groups:
            n_op_types = iterator_shape.embedding_input_shape["type"]
        params_dict = {
            "op_groups": op_groups,
            "n_op_types": n_op_types,
            "input_size": input_size,
            **kwargs,
        }
        if any((name not in cls.Params.__dataclass_fields__) for name in params_dict):
            for name in params_dict:
                if name not in cls.Params.__dataclass_fields__:
                    logger.debug(f"{name} is not a valid parameter for {cls.__name__}")
            raise ValueError(f"Some parameters are not valid for {cls.__name__} Params")
        return cls(cls.Params(**params_dict))

    def __init__(self, net_params: Params) -> None:
        super().__init__(net_params)
        self.input_size = net_params.input_size

        op_groups = net_params.op_groups
        self.op_type = "type" in op_groups
        self.op_cbo = "cbo" in op_groups
        self.op_enc = "op_enc" in op_groups
        if self.op_type:
            if net_params.n_op_types is None or net_params.type_embedding_dim is None:
                raise ValueError(
                    "n_op_types and type_embedding_dim must be provided "
                    "if `type` is included in op_groups"
                )
            self.op_embedder = nn.Embedding(
                net_params.n_op_types, net_params.type_embedding_dim
            )
            self.input_size += net_params.type_embedding_dim
        self.out_norm: Optional[nn.Module] = None
        if net_params.embedding_normalizer is None:
            self.out_norm = None
        elif net_params.embedding_normalizer == "BN":
            self.out_norm = nn.BatchNorm1d(self.embedding_size)
        elif net_params.embedding_normalizer == "LN":
            self.out_norm = nn.LayerNorm(self.embedding_size)
        elif net_params.embedding_normalizer == "IsoBN":
            self.out_norm = IsoBN(self.embedding_size)
        else:
            raise ValueError(net_params.embedding_normalizer)

    def concatenate_op_features(self, g: dgl.DGLGraph) -> th.Tensor:
        """Concatenate the operation features into a single tensor.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input graph

        Returns
        -------
        th.Tensor
            output tensor of shape (num_nodes, input_size)
        """
        op_list = []
        if self.op_type:
            op_list.append(self.op_embedder(g.ndata["op_gid"]))
        if self.op_cbo:
            op_list.append(g.ndata["cbo"])
        if self.op_enc:
            op_list.append(g.ndata["op_enc"])
        op_tensor = th.cat(op_list, dim=1) if len(op_list) > 1 else op_list[0]
        return op_tensor

    def normalize_embedding(self, embedding: th.Tensor) -> th.Tensor:
        """Normalizes the embedding."""
        if self.out_norm is not None:
            embedding = self.out_norm(embedding)
        return embedding
