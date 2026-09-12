"""The GNN itself: a small stack of graph layers for node-role classification.

Two architectures, selected by ``GNNConfig.architecture``:

- ``"gcn"`` (default, ``SeizureGCN``) -- plain ``GCNConv`` (Kipf & Welling),
  every neighbour aggregated with a fixed, degree-normalized weight. This is
  what ``gnn_model_result/baseline_overfit/`` and ``.../regularized/`` were
  trained with.
- ``"gat"`` (``SeizureGAT``) -- ``GATv2Conv`` (Brody et al. 2021), which
  *learns* how much to trust each neighbour instead of using a fixed
  normalization, and folds each edge's own measured weight (recruitment
  latency / co-activation correlation) into that attention score via
  ``edge_dim=1`` rather than baking it into a fixed coefficient.

  More heads/layers is more capacity, not automatically more regularization
  -- ``heads=4, num_layers=2`` (1994 parameters, 9x ``SeizureGCN``'s own)
  overfits *faster and less stably* than the GCN baseline on this 97-node
  graph (see ``gnn_model.run_gnn``'s module docstring). The trained
  ``gnn_model_result/attention/`` comparison uses ``heads=1, num_layers=1``
  instead -- a single attention head mapping features straight to class
  logits, 54 parameters, smaller than the GCN baseline -- which is enough
  for the attention mechanism itself to help (learned, edge-weight-aware
  neighbour trust) without the extra heads/layers reintroducing the same
  instability they were meant to fix.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv

__all__ = ["GNNConfig", "SeizureGCN", "SeizureGAT"]


@dataclass
class GNNConfig:
    hidden_channels: int = 16
    num_layers: int = 2
    dropout: float = 0.5
    seed: int = 7
    architecture: str = "gcn"
    heads: int = 4  # SeizureGAT only


class SeizureGCN(nn.Module):
    """``num_layers`` stacked ``GCNConv``s; ReLU + dropout between layers, raw logits out."""

    def __init__(self, in_channels: int, out_channels: int, config: GNNConfig):
        super().__init__()
        if config.num_layers < 1:
            raise ValueError("GNNConfig.num_layers must be >= 1")
        self.config = config
        if config.num_layers == 1:
            sizes = [in_channels, out_channels]
        else:
            sizes = [in_channels] + [config.hidden_channels] * (config.num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(GCNConv(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
               edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.config.dropout, training=self.training)
        return x

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        """Plain-language statement of this model's own architecture -- the "ГНС параметры" report."""
        lines = [f"SeizureGCN: {len(self.convs)} GCNConv layer(s), "
                f"hidden_channels={self.config.hidden_channels}, dropout={self.config.dropout}, "
                f"seed={self.config.seed}"]
        for i, conv in enumerate(self.convs):
            layer_params = sum(p.numel() for p in conv.parameters())
            lines.append(f"  layer {i + 1}: GCNConv({conv.in_channels} -> {conv.out_channels}), "
                        f"{layer_params} parameters")
        lines.append(f"  total trainable parameters: {self.num_parameters()}")
        return "\n".join(lines)


class SeizureGAT(nn.Module):
    """``num_layers`` stacked ``GATv2Conv``s -- learned per-neighbour attention instead of
    ``SeizureGCN``'s fixed degree normalization; ELU + dropout between layers, raw logits out.

    Every layer also receives the edge's own weight (the same one ``SeizureGCN`` uses as a
    fixed multiplier) as a 1-dim edge feature the attention score can up- or down-weight per
    node, rather than trusting it uniformly. Hidden layers concatenate all heads
    (``concat=True``, so their width is ``hidden_channels * heads``); the output layer averages
    them (``concat=False``) down to the class logits.
    """

    def __init__(self, in_channels: int, out_channels: int, config: GNNConfig):
        super().__init__()
        if config.num_layers < 1:
            raise ValueError("GNNConfig.num_layers must be >= 1")
        if config.heads < 1:
            raise ValueError("GNNConfig.heads must be >= 1")
        self.config = config
        heads = config.heads
        hidden = config.hidden_channels

        self.convs = nn.ModuleList()
        if config.num_layers == 1:
            # heads>1 still applies here (averaged, concat=False): multiple attention heads voting
            # on the same in -> out logits is a real ensemble effect even with a single layer.
            self.convs.append(GATv2Conv(in_channels, out_channels, heads=heads, concat=False,
                                        dropout=config.dropout, edge_dim=1))
        else:
            self.convs.append(GATv2Conv(in_channels, hidden, heads=heads, concat=True,
                                        dropout=config.dropout, edge_dim=1))
            width = hidden * heads
            for _ in range(config.num_layers - 2):
                self.convs.append(GATv2Conv(width, hidden, heads=heads, concat=True,
                                            dropout=config.dropout, edge_dim=1))
                width = hidden * heads
            self.convs.append(GATv2Conv(width, out_channels, heads=1, concat=False,
                                        dropout=config.dropout, edge_dim=1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
               edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        edge_attr = edge_weight.unsqueeze(-1) if edge_weight is not None else None
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.config.dropout, training=self.training)
        return x

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        lines = [f"SeizureGAT: {len(self.convs)} GATv2Conv layer(s), "
                f"hidden_channels={self.config.hidden_channels}, heads={self.config.heads}, "
                f"dropout={self.config.dropout}, seed={self.config.seed}"]
        for i, conv in enumerate(self.convs):
            layer_params = sum(p.numel() for p in conv.parameters())
            out_width = conv.out_channels * (conv.heads if conv.concat else 1)
            combine = "concat" if conv.concat else "averaged"
            lines.append(f"  layer {i + 1}: GATv2Conv({conv.in_channels} -> {conv.out_channels} x "
                        f"{conv.heads} head(s), {combine} -> {out_width} dim(s)), {layer_params} parameters")
        lines.append(f"  total trainable parameters: {self.num_parameters()}")
        return "\n".join(lines)
