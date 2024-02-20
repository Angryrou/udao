from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Tuple, Type

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn


def e_src_dot_dst(
    src_field: str, dst_field: str, out_field: str
) -> Callable[[Any], Dict[str, torch.Tensor]]:
    """Multiply source and destination node features and sum them up"""

    def func(edges: Any) -> Dict[str, torch.Tensor]:
        return {
            out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(
                -1, keepdim=True
            )
        }

    return func


def e_scaled_exp(
    field: str, scale_constant: float
) -> Callable[[Any], Dict[str, torch.Tensor]]:
    """Compute scaled exponential of a graph's edge field"""

    def func(edges: Any) -> Dict[str, torch.Tensor]:
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Attention Layer for Graph
    proposed by "A Generalization of Transformer Networks to Graphs", DLG-AAAI'21.
    https://arxiv.org/pdf/2012.09699.pdf

    Parameters
    ----------
        in_dim : int
            Input dimension
        out_dim : int
            Output dimension
        n_heads : int
            Number of attention heads
        use_bias : bool
            Whether to use bias
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int, use_bias: bool) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads

        self.Q = nn.Linear(in_dim, out_dim * n_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * n_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * n_heads, bias=use_bias)

    def _apply_attention(self, g: dgl.DGLGraph) -> torch.Tensor:
        edge_ids: Tuple[torch.Tensor, torch.Tensor] = g.edges()
        g.send_and_recv(
            edge_ids,
            fn.u_mul_e("V_h", "score", "V_h"),  # type: ignore
            fn.sum("V_h", "wV"),  # type: ignore
        )
        g.send_and_recv(
            edge_ids, fn.copy_e("score", "score"), fn.sum("score", "z")  # type: ignore
        )
        head_out = g.ndata["wV"] / (
            g.ndata["z"] + torch.full_like(g.ndata["z"], 1e-6)  # type: ignore
        )
        # uncommenting below would delete intermediate values from graph.
        # Necessary if hidden_dim is different from out_dim
        # g.ndata.pop("wV")
        # g.ndata.pop("z")
        # g.edata.pop("score")
        return head_out

    def compute_attention(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """Simple attention mechanism"""
        g.apply_edges(e_src_dot_dst("K_h", "Q_h", "score"))
        g.apply_edges(e_scaled_exp("score", np.sqrt(self.out_dim)))
        return g

    def compute_query_key_value(self, g: dgl.DGLGraph, h: torch.Tensor) -> dgl.DGLGraph:
        """Compute query, key, and value for each node"""
        g.ndata["Q_h"] = self.Q(h).view(-1, self.n_heads, self.out_dim)
        g.ndata["K_h"] = self.K(h).view(-1, self.n_heads, self.out_dim)
        g.ndata["V_h"] = self.V(h).view(-1, self.n_heads, self.out_dim)
        return g

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        g = self.compute_query_key_value(g, h)
        g = self.compute_attention(g)
        return self._apply_attention(g)


class RAALMultiHeadAttentionLayer(MultiHeadAttentionLayer):
    """MultiHead Attention using Resource-Aware Attentional LSTM
    proposed by "A Resource-Aware Deep Cost Model for Big Data Query Processing”
    https://ieeexplore.ieee.org/document/9835426

    The RAAL MultiHead Attention Layer requires the graphs to have an "sid" node
    feature, corresponding to the id of the template graph.
    This makes the link with the non_siblings_map.

    Parameters
    ----------
    non_siblings_map : Dict[int, Dict[int, List[int]]]
    For each type of graph, maps the edge id to
    all nodes that are not siblings of its source node
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int,
        use_bias: bool,
        non_siblings_map: Dict[int, Dict[int, List[int]]],
    ) -> None:
        super().__init__(in_dim, out_dim, n_heads, use_bias)
        self.non_siblings_map = non_siblings_map

    def compute_attention(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """Attention mechanism with non-siblings attention"""
        g_list = dgl.unbatch(g)
        sid_g_map: Dict[int, List[dgl.DGLGraph]] = defaultdict(list)
        sid_g_batch_ids: Dict[int, List[int]] = defaultdict(list)

        for gid, gg in enumerate(g_list):
            sid = gg.ndata["sid"][0].detach().cpu().item()
            sid_g_map[sid].append(gg)
            sid_g_batch_ids[sid].append(gid)

        sid_gb_map: Dict[int, dgl.DGLGraph] = defaultdict(list)
        for sid, graphs in sid_g_map.items():
            gb = dgl.batch(graphs)
            n_graphs = len(graphs)
            Q = gb.ndata["Q_h"].reshape(n_graphs, -1, self.n_heads, self.out_dim)  # type: ignore
            K = gb.ndata["K_h"].reshape(n_graphs, -1, self.n_heads, self.out_dim)  # type: ignore
            QK = (
                torch.matmul(Q.transpose(1, 2), K.transpose(1, 2).transpose(2, 3))
                .transpose(1, 2)
                .clamp(-5, 5)
            )
            srcs, dsts, eids = graphs[0].edges(form="all", order="srcdst")
            score_list = [
                QK[:, src, :, dst]
                / (
                    QK[:, src, :, self.non_siblings_map[sid][eid]].sum(-1)
                    + torch.full_like(QK[:, src, :, dst], 1e-6)
                )
                for src, dst, eid in zip(
                    srcs.cpu().numpy(), dsts.cpu().numpy(), eids.cpu().numpy()
                )
            ]
            gb.edata["score"] = torch.cat(score_list, dim=1).view(-1, self.n_heads, 1)
            sid_gb_map[sid] = gb

        for sid, gb in sid_gb_map.items():
            g_batch_inds = sid_g_batch_ids[sid]
            for batch_id, new_g in zip(g_batch_inds, dgl.unbatch(gb)):
                g_list[batch_id] = new_g
        return dgl.batch(g_list)


class QFMultiHeadAttentionLayer(MultiHeadAttentionLayer):
    """MultiHead Attention using QueryFormer
    proposed by "QueryFormer: A Tree Transformer Model for Query Plan
    Representation"
    https://www.vldb.org/pvldb/vol15/p1658-zhao.pdf

    The QF MultiHead Attention Layer requires the graphs to have a "dist"
    edge feature. see examples/data/spark/4.qf_addition

    Parameters
    ----------
    attention_bias : torch.Tensor
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int,
        use_bias: bool,
        attention_bias: torch.Tensor,
    ) -> None:
        super().__init__(in_dim, out_dim, n_heads, use_bias)
        self.attention_bias = attention_bias

    def compute_attention(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """Attention mechanism with attention bias"""
        g = super().compute_attention(g)
        g.edata["att_bias"] = torch.index_select(
            self.attention_bias, 0, g.edata["dist"] - 1  # type: ignore
        ).reshape(-1, 1, 1)
        g.edata["score"] = g.edata["score"] + g.edata["att_bias"]  # type: ignore
        return g


AttentionLayerName = Literal["QF", "GTN", "RAAL"]

ATTENTION_TYPES: Dict[AttentionLayerName, Type[MultiHeadAttentionLayer]] = {
    "QF": QFMultiHeadAttentionLayer,
    "GTN": MultiHeadAttentionLayer,
    "RAAL": RAALMultiHeadAttentionLayer,
}
