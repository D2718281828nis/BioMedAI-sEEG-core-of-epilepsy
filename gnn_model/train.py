"""Full-batch training loop for a ``SeizureGCN``/``SeizureGAT`` node classifier on one
transductive node-classification graph.

There is exactly one graph here (one EDF recording), so this is ordinary
semi-supervised node classification (a la Kipf & Welling's GCN on Cora):
every node is visible to every forward pass, only the loss/metrics are
masked to ``dataset.data.train_mask`` / ``val_mask``. No mini-batching, no
neighbour sampling -- the graph (under 100 nodes) fits in a single forward
pass many times over.

With ``drop_edge_p=0.0`` and ``early_stopping_patience=None`` (both
defaults) this reproduces the original, deliberately-unregularized baseline
bit-for-bit (no extra randomness is drawn, no early exit) -- that run is the
whole point of ``gnn_model_result/baseline_overfit/``: with only 5
"earliest" positives in the entire graph, a plain GCN memorizes them
(train_accuracy keeps climbing) while ``val_loss`` diverges, which is the
demonstration this pipeline exists to produce, not a bug to hide. Passing
``drop_edge_p>0`` and/or ``early_stopping_patience`` is the *other*, safe way
to fight that same overfitting -- see ``gnn_model.run_gnn``'s module
docstring for both.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from .data import GraphDataset
from .model import GNNConfig, SeizureGAT, SeizureGCN

__all__ = ["TrainingResult", "train_gnn"]

_ARCHITECTURES = {"gcn": SeizureGCN, "gat": SeizureGAT}


@dataclass
class TrainingResult:
    model: nn.Module
    history: dict[str, list[float]]
    train_confusion_matrix: np.ndarray
    val_confusion_matrix: np.ndarray
    train_report: dict = field(repr=False)
    val_report: dict = field(repr=False)
    class_names: list[str]
    epochs: int
    best_epoch: int | None = None
    stopped_early: bool = False
    early_stopping_metric: str | None = None


def _class_weights(y: torch.Tensor, mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights over the *train* split, so 92-vs-5 imbalance doesn't just teach
    the model to always predict the majority role."""
    counts = np.bincount(y[mask].numpy(), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _drop_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor, p: float) -> tuple[torch.Tensor, torch.Tensor]:
    """DropEdge (Rong et al. 2020): randomly discard a fraction ``p`` of edge entries each
    training step, so the model can't just wire around 97 nodes' worth of fixed neighbourhoods.

    Each directed entry (the graph is stored as both (i, j) and (j, i)) is dropped
    independently -- simpler than pairing them back up, and still a valid stochastic
    regularizer since GCNConv has no requirement that its input be symmetric.
    """
    if p <= 0:
        return edge_index, edge_weight
    keep = torch.rand(edge_index.size(1)) >= p
    if not bool(keep.any()):
        keep[torch.randint(0, edge_index.size(1), (1,))] = True
    return edge_index[:, keep], edge_weight[keep]


def train_gnn(dataset: GraphDataset, config: GNNConfig | None = None, epochs: int = 200,
             lr: float = 0.01, weight_decay: float = 5e-4, drop_edge_p: float = 0.0,
             early_stopping_patience: int | None = None,
             early_stopping_metric: str = "val_loss") -> TrainingResult:
    """Train the architecture named by ``config.architecture`` (``"gcn"`` -> ``SeizureGCN``,
    ``"gat"`` -> ``SeizureGAT``; see ``gnn_model.model``) on ``dataset``.

    ``drop_edge_p`` (0 by default) and ``early_stopping_patience`` (``None``
    by default, i.e. off) are training-level knobs for fighting overfitting;
    switching ``config.architecture`` to ``"gat"`` is the structural one --
    see this module's docstring and ``gnn_model.model``'s. Leaving
    ``config`` at ``GNNConfig()`` and both knobs off reproduces the
    unregularized GCN baseline exactly.

    ``early_stopping_metric`` ("val_loss", the default, or "val_accuracy")
    picks what "best" means for the restored checkpoint. "val_loss" is the
    textbook choice, but with a heavily class-weighted loss on 2-3 train
    positives, its very first few epochs (near-random logits, weighted
    almost arbitrarily by the class prior) can look "best" before the model
    has learned anything -- SeizureGAT in particular keeps climbing from
    epoch 1 with these class weights (see ``gnn_model_result/attention/``).
    "val_accuracy" is not this fragile to that early-training noise and is
    the metric used for that run.
    """
    config = config or GNNConfig()
    torch.manual_seed(config.seed)
    data = dataset.data
    num_classes = len(dataset.class_names)

    model_cls = _ARCHITECTURES.get(config.architecture)
    if model_cls is None:
        raise ValueError(f"Unknown GNNConfig.architecture {config.architecture!r} -- "
                         f"choose from {sorted(_ARCHITECTURES)}")
    if early_stopping_metric not in ("val_loss", "val_accuracy"):
        raise ValueError(f"early_stopping_metric must be 'val_loss' or 'val_accuracy', "
                         f"got {early_stopping_metric!r}")
    higher_is_better = early_stopping_metric == "val_accuracy"

    model = model_cls(in_channels=data.x.shape[1], out_channels=num_classes, config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weight = _class_weights(data.y, data.train_mask, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    history: dict[str, list[float]] = {"loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_metric = float("-inf") if higher_is_better else float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_edge_index, train_edge_weight = _drop_edges(data.edge_index, data.edge_weight, drop_edge_p)
        out = model(data.x, train_edge_index, train_edge_weight)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_weight)
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            train_pred = out[data.train_mask].argmax(dim=1)
            val_pred = out[data.val_mask].argmax(dim=1)
            train_acc = (train_pred == data.y[data.train_mask]).float().mean().item()
            val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()

        history["loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        if early_stopping_patience is not None:
            current = val_acc if higher_is_better else val_loss.item()
            improved = current > best_metric if higher_is_better else current < best_metric
            if improved:
                best_metric = current
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    stopped_early = True
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight)
        train_pred = out[data.train_mask].argmax(dim=1).numpy()
        val_pred = out[data.val_mask].argmax(dim=1).numpy()
    train_true = data.y[data.train_mask].numpy()
    val_true = data.y[data.val_mask].numpy()
    labels = list(range(num_classes))

    train_cm = confusion_matrix(train_true, train_pred, labels=labels)
    val_cm = confusion_matrix(val_true, val_pred, labels=labels)
    train_report = classification_report(train_true, train_pred, labels=labels,
                                         target_names=dataset.class_names, output_dict=True, zero_division=0)
    val_report = classification_report(val_true, val_pred, labels=labels,
                                       target_names=dataset.class_names, output_dict=True, zero_division=0)

    return TrainingResult(model=model, history=history, train_confusion_matrix=train_cm,
                          val_confusion_matrix=val_cm, train_report=train_report, val_report=val_report,
                          class_names=dataset.class_names, epochs=len(history["loss"]),
                          best_epoch=best_epoch, stopped_early=stopped_early,
                          early_stopping_metric=early_stopping_metric if best_epoch is not None else None)
