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

``train_gnn`` pins its thread pools (``torch.set_num_threads(1)``,
``torch.set_num_interop_threads(1)`` where settable, and
``threadpoolctl.threadpool_limits(1)`` for the native OpenMP/BLAS pool
underneath both) before training. This makes every result in this package
reproducible run to run *within one Python environment* (verified
repeatedly, single-split and cross-validated alike) -- but it does not, and
cannot, make results agree *across* environments built against different
BLAS libraries: on this same machine, the identical command under a
NumPy/PyTorch built against OpenBLAS (Anaconda's default) versus one built
against Apple's Accelerate framework produces different floating-point
rounding in the same matrix operations, and on ``SeizureGAT``'s deeper,
small/uneven-fold cross-validation runs specifically, that was enough to
flip which local optimum a fold's training lands on -- e.g.
``gnn_model_result/attention_deep_dfa_cv_shaft/`` (seed 7) reads
``[0,5]``/``[2,90]`` under OpenBLAS and ``[3,2]``/``[3,89]`` under
Accelerate for byte-identical inputs and code. Plain ``StratifiedKFold``
runs on this same graph were not observed to have this sensitivity
(bit-identical across both backends, every configuration tried); only
``--group-by-shaft``'s smaller, unevenly-sized folds were fragile enough to
show it, and separately (see ``gnn_model.run_gnn``'s module docstring for
the full numbers) that same configuration is at least as sensitive to
*random seed* as to BLAS backend -- treat any single shaft-grouped,
DFA-augmented result as one draw from a wide distribution, not a
measurement, and prefer the multi-seed summary over the single saved
seed-7 run when the question is "does this generalize" rather than "what
does this exact command print." ``SeizureGCN`` and plain-fold
cross-validation were never observed to have either sensitivity.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn

from .data import GraphDataset
from .model import GNNConfig, SeizureGAT, SeizureGCN

__all__ = ["TrainingResult", "train_gnn", "CrossValidationResult", "cross_validate_gnn"]

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
             early_stopping_metric: str = "val_loss", label_smoothing: float = 0.0) -> TrainingResult:
    """Train the architecture named by ``config.architecture`` (``"gcn"`` -> ``SeizureGCN``,
    ``"gat"`` -> ``SeizureGAT``; see ``gnn_model.model``) on ``dataset``.

    ``drop_edge_p`` (0 by default) and ``early_stopping_patience`` (``None``
    by default, i.e. off) are training-level knobs for fighting overfitting;
    switching ``config.architecture`` to ``"gat"`` is the structural one --
    see this module's docstring and ``gnn_model.model``'s. Leaving
    ``config`` at ``GNNConfig()`` and both knobs off reproduces the
    unregularized GCN baseline exactly.

    ``early_stopping_metric`` picks what "best" means for the restored
    checkpoint:

    - ``"val_loss"`` (default): the textbook choice, but with a heavily
      class-weighted loss on 2-3 train positives, its very first few epochs
      (near-random logits, weighted almost arbitrarily by the class prior)
      can look "best" before the model has learned anything -- SeizureGAT in
      particular keeps climbing from epoch 1 with these class weights.
    - ``"val_accuracy"``: not fragile to that early-training noise, but on a
      92:5 imbalance a heavily-regularized model can raise accuracy simply
      by *always* predicting the majority role -- accuracy alone can't tell
      that apart from a model that has actually learned to separate the
      classes (both score ~0.93 here; only the confusion matrix shows the
      difference).
    - ``"val_macro_f1"``: unweighted mean of per-class F1, so a checkpoint
      that ignores the minority class scores 0.5 at best regardless of how
      accurate it looks overall -- the metric actually used for
      ``gnn_model_result/attention/``.

    ``val_loss`` climbing even while ``val_accuracy``/``val_macro_f1`` hold
    steady (see ``gnn_model_result/attention_deep/``'s early runs) is
    calibration drift, not a changing decision boundary: with this small a
    graph the optimizer keeps growing weight/logit magnitudes long after the
    predictions themselves have stopped changing, and a class-weighted
    ``CrossEntropyLoss`` charges an ever-larger penalty for the wrong
    predictions still being made, ever more confidently. Two knobs curb
    that directly, and are not equivalent: **``weight_decay``** (L2 on the
    weights themselves) controls the root cause and, at
    ``gnn_model_result/attention_deep/``'s architecture, raising it from
    5e-3 to 0.2 turns ``val_loss`` from *diverging* (2.75 max) into
    *converging* (plateaus at ~1.0-1.08 for 100+ epochs) with the *same*
    confusion matrix -- see this module's own training curve, plotted in
    that run's ``gnn_loss_curve.png``. **``label_smoothing``** (0 by
    default here) caps the loss indirectly, by capping how confident any
    target is allowed to be (so cross-entropy can never charge the full
    penalty for a wrong answer regardless of the weights); it does reduce
    ``val_loss``'s ceiling, but on this graph it also pushed the optimizer
    toward a worse decision boundary at every value tried (more
    "later_recruited" nodes misclassified) -- a real trade-off, not a free
    win, unlike raising ``weight_decay`` here.
    """
    # Belt-and-braces: torch.set_num_threads(1) alone did *not* reliably fix the divergence this
    # module's docstring describes (empirically -- it governs torch's own intraop scheduler, not
    # the OpenMP thread pool torch's compiled CPU kernels dispatch into, which is what actually
    # drives the non-deterministic reduction order). threadpoolctl.threadpool_limits pins that
    # OpenMP pool (and any BLAS backend) at runtime regardless of when/how it was initialized.
    import threadpoolctl
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already fixed by an earlier call in this process (only settable once) -- fine,
              # it can only have been fixed to 1 by this same guard.
    with threadpoolctl.threadpool_limits(limits=1):
        return _train_gnn_single_threaded(dataset, config, epochs, lr, weight_decay, drop_edge_p,
                                          early_stopping_patience, early_stopping_metric, label_smoothing)


def _train_gnn_single_threaded(dataset: GraphDataset, config: GNNConfig | None, epochs: int, lr: float,
                               weight_decay: float, drop_edge_p: float, early_stopping_patience: int | None,
                               early_stopping_metric: str, label_smoothing: float) -> TrainingResult:
    """The actual training loop -- only ever called from ``train_gnn``, which wraps it in a
    single-threaded ``threadpoolctl`` context first (see that function's comment)."""
    config = config or GNNConfig()
    torch.manual_seed(config.seed)
    data = dataset.data
    num_classes = len(dataset.class_names)

    model_cls = _ARCHITECTURES.get(config.architecture)
    if model_cls is None:
        raise ValueError(f"Unknown GNNConfig.architecture {config.architecture!r} -- "
                         f"choose from {sorted(_ARCHITECTURES)}")
    if early_stopping_metric not in ("val_loss", "val_accuracy", "val_macro_f1"):
        raise ValueError("early_stopping_metric must be 'val_loss', 'val_accuracy' or "
                         f"'val_macro_f1', got {early_stopping_metric!r}")
    higher_is_better = early_stopping_metric in ("val_accuracy", "val_macro_f1")

    model = model_cls(in_channels=data.x.shape[1], out_channels=num_classes, config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weight = _class_weights(data.y, data.train_mask, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)

    history: dict[str, list[float]] = {"loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [],
                                       "val_macro_f1": []}
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
            val_macro_f1 = f1_score(data.y[data.val_mask].numpy(), val_pred.numpy(),
                                    labels=list(range(num_classes)), average="macro", zero_division=0)

        history["loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        if early_stopping_patience is not None:
            current = {"val_loss": val_loss.item(), "val_accuracy": val_acc,
                      "val_macro_f1": val_macro_f1}[early_stopping_metric]
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


def _confusion_matrix_report(cm: np.ndarray, class_names: list[str]) -> dict:
    """Per-class precision/recall/f1/support plus macro averages, computed directly from a
    confusion matrix -- no raw prediction arrays needed, so this works just as well on a
    *summed* out-of-fold matrix (``cross_validate_gnn``) as on a single fold's."""
    report: dict[str, dict[str, float]] = {}
    precisions, recalls, f1s = [], [], []
    for i, name in enumerate(class_names):
        support = int(cm[i, :].sum())
        predicted = int(cm[:, i].sum())
        true_positive = int(cm[i, i])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[name] = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    report["macro avg"] = {"precision": float(np.mean(precisions)), "recall": float(np.mean(recalls)),
                           "f1-score": float(np.mean(f1s)), "support": int(cm.sum())}
    return report


@dataclass
class CrossValidationResult:
    """Aggregate of one ``train_gnn`` run per fold of a ``StratifiedKFold`` split.

    ``out_of_fold_confusion_matrix`` is the sum of every fold's own
    ``val_confusion_matrix`` -- since ``load_graph_kfold_datasets`` puts each
    channel node in exactly one fold's validation set, this single matrix
    covers every classifiable node in the graph exactly once, unlike a
    single ``train_gnn`` run's ``val_confusion_matrix`` (~30% of them, from
    one arbitrary split). The ``mean_``/``std_`` fields are computed across
    folds at each fold's own reported checkpoint (its best epoch if early
    stopping was used, its last epoch otherwise).
    """
    fold_results: list[TrainingResult]
    class_names: list[str]
    out_of_fold_confusion_matrix: np.ndarray
    out_of_fold_report: dict = field(repr=False)
    mean_train_loss: float = 0.0
    mean_val_loss: float = 0.0
    std_val_loss: float = 0.0
    mean_val_accuracy: float = 0.0
    std_val_accuracy: float = 0.0
    mean_val_macro_f1: float = 0.0
    std_val_macro_f1: float = 0.0


def cross_validate_gnn(datasets: list[GraphDataset], config: GNNConfig | None = None,
                       **train_gnn_kwargs) -> CrossValidationResult:
    """Run ``train_gnn`` once per fold in ``datasets`` (from ``gnn_model.data.load_graph_kfold_datasets``)
    with identical hyperparameters, and aggregate the results.

    This exists for the same reason ``gnn_model_result/attention_deep/`` needed checking
    epoch-by-epoch before trusting its plateau: with only 5 "earliest" nodes in the whole graph,
    a *single* 70/30 split's confusion matrix depends heavily on which 1-2 of them happened to
    land in the validation set. Cross-validation does not change what any one model learns, but
    it does show whether a result generalizes across *which* nodes get held out, rather than
    reporting one split's number as if it were the only possible one.
    """
    if len(datasets) < 2:
        raise ValueError(f"cross_validate_gnn needs at least 2 folds, got {len(datasets)}")
    class_names = datasets[0].class_names
    num_classes = len(class_names)

    fold_results = [train_gnn(dataset, config=config, **train_gnn_kwargs) for dataset in datasets]

    out_of_fold_cm = np.zeros((num_classes, num_classes), dtype=int)
    for result in fold_results:
        out_of_fold_cm += result.val_confusion_matrix

    def _at_checkpoint(result: TrainingResult, key: str) -> float:
        index = (result.best_epoch - 1) if result.best_epoch is not None else -1
        return result.history[key][index]

    val_losses = [_at_checkpoint(r, "val_loss") for r in fold_results]
    val_accuracies = [_at_checkpoint(r, "val_accuracy") for r in fold_results]
    val_f1s = [_at_checkpoint(r, "val_macro_f1") for r in fold_results]
    train_losses = [_at_checkpoint(r, "loss") for r in fold_results]

    return CrossValidationResult(
        fold_results=fold_results, class_names=class_names, out_of_fold_confusion_matrix=out_of_fold_cm,
        out_of_fold_report=_confusion_matrix_report(out_of_fold_cm, class_names),
        mean_train_loss=float(np.mean(train_losses)),
        mean_val_loss=float(np.mean(val_losses)), std_val_loss=float(np.std(val_losses)),
        mean_val_accuracy=float(np.mean(val_accuracies)), std_val_accuracy=float(np.std(val_accuracies)),
        mean_val_macro_f1=float(np.mean(val_f1s)), std_val_macro_f1=float(np.std(val_f1s)))
