from dataclasses import dataclass

import dgl
import torch as th
import torch.nn as nn

from .base_graph_embedder import BaseGraphEmbedder


class GraphAverager(BaseGraphEmbedder):
    """Averager Embedder network.
    Computes an embedding for each operation using a linear layer,
    then averages the embeddings of all operations in the graph.
    """

    @dataclass
    class Params(BaseGraphEmbedder.Params):
        pass

    def __init__(self, net_params: Params) -> None:
        super().__init__(net_params)

        self.emb = nn.Sequential(
            nn.Linear(self.input_size, self.embedding_size), nn.ReLU()
        )

    def _embed(self, g: dgl.DGLGraph, h: th.Tensor) -> th.Tensor:
        h = self.emb(h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")

    def forward(self, g: dgl.DGLGraph) -> th.Tensor:  # type: ignore[override]
        h = self.concatenate_op_features(g)
        return self.normalize_embedding(self._embed(g, h))
