"""Build and train a graph neural network (GCN) node classifier on an already-built seizure graph.

Run as a module (relative imports inside the package), from the repo root:

    python -m gnn_model.run_gnn --graph object_model_result/sEEG-HFOs-8/object_model_graph.graphml \\
        --output gnn_model_result

Deliberately a *separate* pipeline from ``object_model.run_object_model``:
this one never touches the EDF/DICOM/reservoir data directly, only the
GraphML that pipeline (or ``extreme_event_agent.edf_workflow.build_seizure_graph``
alone) already wrote. It loads that graph, trains a small transductive node
classifier (``gnn_model.model.SeizureGCN``) to predict each channel node's
already-computed ``role`` from its own EDF/structural/reservoir attributes
(see ``gnn_model.data`` for the exact feature schema and why ``is_initiator``
is excluded), and reports the same things any training run should: the
loss/val_loss curves, the confusion matrix on both splits, and the model's
own architecture/parameter count.

Three ways to invoke this, all worth keeping around side by side (see
``gnn_model_result/*/sEEG-HFOs-8/gnn_model_summary.txt`` for the saved runs):

- **Defaults** (``--architecture gcn --drop-edge-p 0 --early-stopping-patience``
  unset) reproduce the *unregularized baseline*: with only 5 "earliest"
  positives in the whole graph, the GCN memorizes them and ``val_loss``
  diverges to **6.29** even as ``val_accuracy`` (0.933) looks fine (accuracy
  hides it; loss doesn't). That divergence is not a bug -- it is the
  pipeline's own demonstration of why a data-hungry method like a plain GNN
  cannot be trusted on genuinely rare events, which is exactly the argument
  for the classical, few-sample-valid statistics (extreme value theory)
  ``extreme_event_agent`` is named after. See ``gnn_model_result/
  baseline_overfit/`` (and its ``EVT_PROOF_small_sample_overfitting.md``).
- **``--drop-edge-p 0.3 --early-stopping-patience 20`` (+ a smaller/more
  regularized GCN)** fights the same overfitting at the *training* level:
  DropEdge (Rong et al. 2020) randomly discards edges each step so the model
  can't wire around this small a neighbourhood structure, and early
  stopping restores the checkpoint with the best ``val_loss`` instead of
  the final, overfit one. It keeps ``val_loss`` bounded (0.66 vs. 6.29 --
  ``gnn_model_result/regularized/``), but stopping on the very first
  ``val_loss`` dip just halts training before the model has separated the
  classes at all: it stops at epoch 7 of 27, and validation accuracy at
  that checkpoint is only 0.333 (19 of 28 "later_recruited" nodes wrongly
  called "earliest").
- **``--architecture gat``** is the *structural* answer to that same
  problem: ``SeizureGAT`` (``GATv2Conv``, Brody et al. 2021, see
  ``gnn_model.model``) learns per-neighbour attention instead of GCN's fixed
  degree-normalized aggregation, and folds each edge's own measured weight
  into that attention score rather than trusting it uniformly. A first pass
  at this (``--heads 4 --num-layers 2``, 1994 parameters -- 9x the GCN
  baseline) just overfit *again*, faster and less predictably than the GCN
  did: ``val_loss`` climbed from 0.70 to over 2.0 within 15 epochs and
  ``train_accuracy`` oscillated between 0.54 and 0.93 run to run instead of
  climbing smoothly. More attention heads and more layers is more capacity,
  not automatically more regularization. The fix was to *right-size* it:
  ``--heads 1 --num-layers 1`` is a single attention head mapping features
  straight to class logits -- **54 parameters, smaller than the GCN
  baseline itself** -- combined with ``--drop-edge-p 0.2`` and
  checkpointing on ``--early-stopping-metric val_macro_f1`` rather than
  ``val_loss`` or ``val_accuracy`` (this class-weighted loss looks
  deceptively "best" in ``SeizureGAT``'s very first epoch and only climbs
  from there, and on this 92:5 imbalance raw accuracy can't tell a model
  that has actually learned the minority class apart from one that just
  always predicts the majority -- both score ~0.93; see
  ``gnn_model.train``'s docstring). Combined (``--architecture gat --heads 1
  --num-layers 1 --drop-edge-p 0.2 --early-stopping-patience 40
  --early-stopping-metric val_macro_f1``) this keeps ``val_loss`` bounded
  (0.79 at the checkpoint vs. 6.29 for the baseline and vs. climbing past
  2.0 for the oversized GAT) with only 3 of 28 "later_recruited" nodes
  misclassified (vs. 19 for the DropEdge-only GCN) -- see
  ``gnn_model_result/attention/``.
- **More message passing (deeper) with attention, done safely**:
  ``--architecture gat --num-layers 3 --hidden-channels 8 --heads 2
  --residual --drop-edge-p 0.2 --weight-decay 0.2
  --early-stopping-patience 60 --early-stopping-metric val_macro_f1``. Three
  ``GATv2Conv`` layers means three hops of message passing across the
  recruitment/co-activation edges instead of one; ``--residual`` passes
  ``GATv2Conv``'s own learnable skip connection to every layer (without it,
  a plain 3-4 layer stack on a graph this small/low-diameter oversmooths --
  every node's representation collapses toward its neighbourhood average
  before training gets anywhere); hidden layers keep the original
  concatenated-heads convention (``concat_heads=True``, the default), so
  1562 parameters total -- more than the single-layer ``attention/`` run,
  but for a materially better result, not just more capacity for its own
  sake. An early pass at this used ``--weight-decay 5e-3``: checkpointed at
  epoch 22 of 62, ``loss=0.29``, ``val_loss=1.58`` -- bounded relative to the
  baseline's 6.29, but still climbing past 2.75 by the time training
  stopped. Raising ``--weight-decay`` to 0.2 (still just L2 on the weights,
  no architecture change) fixed that directly: checkpointed at epoch 70 of
  130, same confusion matrices (train ``[3,0]``/``[8,56]``, val
  ``[1,1]``/``[1,27]`` -- only one misclassified node per split, matching
  the reckless baseline's own best-case numbers), but ``val_loss`` now
  *plateaus* at ~1.0-1.08 for 100+ epochs instead of climbing indefinitely
  -- see ``gnn_model.train``'s docstring for why weight decay (controls the
  root cause: growing weight/logit magnitudes) beats ``--label-smoothing``
  (controls it indirectly, by capping confidence, and traded away
  confusion-matrix quality at every value tried on this graph) here. See
  ``gnn_model_result/attention_deep/``.
- **Cross-validation, because one split's confusion matrix can be lucky**:
  add ``--cross-validate 5`` to any of the invocations above (with only 5
  "earliest" nodes total, 5 folds means each one holds out almost exactly
  one). Running it on the ``weight_decay=0.2`` configuration above is
  sobering: the *out-of-fold* confusion matrix -- every classifiable node
  counted exactly once, pooled across all 5 folds' own held-out sets --
  reads ``[4,1]`` / ``[21,71]`` (earliest / later_recruited rows), with mean
  ``val_macro_f1=0.55 +/- 0.06`` across folds. That is a materially worse
  picture than the single-split ``[1,1]``/``[1,27]`` above: this
  architecture recalls the rare class reasonably well (4 of 5 "earliest"
  nodes caught somewhere across the 5 folds) but at a real precision cost
  (21 false positives) the one lucky 70/30 split never surfaced, because it
  happened to hold out an easy val subset. Per-fold curves also show 4 of 5
  folds converge cleanly while one diverges toward ``val_loss~1.6`` -- the
  same architecture is not uniformly stable across *which* nodes get held
  out. See ``gnn_model_result/attention_deep_cv/`` and
  ``gnn_model.train.cross_validate_gnn``'s docstring.
- **A new feature (DFA), and "smart" folding to check whether it's real**:
  ``gnn_model.augment_dfa`` computes each channel's Detrended Fluctuation
  Analysis scaling exponent (``gnn_model.dfa``) over its own 30 s pre-event
  baseline and writes it onto the graph as ``dfa_alpha`` -- the one place in
  this package that reads raw EDF samples (see that module's docstring for
  why it stays separate). On this recording, "earliest" channels' exponents
  are tighter and higher (mean 1.22, std 0.05) than "later_recruited"'s
  (mean 1.15, std 0.10; Mann-Whitney p=0.014) -- a real, checkable signal,
  not noise. Feeding it to the ``weight_decay=0.2`` ``attention_deep``
  config and cross-validating (``--graph .../object_model_graph_dfa.graphml
  --cross-validate 5``) looks like a clean win under plain
  ``StratifiedKFold``: out-of-fold confusion matrix improves from
  ``[4,1]``/``[21,71]`` (no DFA) to ``[4,1]``/``[4,88]`` (with DFA),
  ``val_macro_f1`` from 0.55 to 0.84, and this exact result is bit-identical
  across BLAS backends (verified on both an OpenBLAS- and an
  Accelerate-linked NumPy/PyTorch build on the same machine) -- a genuinely
  stable finding at seed 7.
  ``--group-by-shaft`` (``gnn_model.data.load_graph_kfold_datasets``'s
  ``StratifiedGroupKFold`` grouping, so a validation node's own same-shaft
  neighbours -- its strongest co-activation edges -- never sit in train)
  tells a different story, and a more fragile one: at seed 7 specifically,
  the ``+DFA`` out-of-fold matrix reads ``[0,5]``/``[2,90]`` (f1=0.48) versus
  ``[4,1]``/``[19,73]`` (f1=0.50) without DFA -- the same "recall
  evaporates, false-alarm reduction holds" pattern the plain-fold comparison
  above hinted was leakage-dependent. But unlike the plain-fold result,
  *this specific number is not stable*: it changes with the BLAS backend
  (``[3,2]``/``[3,89]``, f1=0.62, on the same seed under Accelerate instead
  of OpenBLAS -- shaft-grouping produces small, unevenly-sized,
  early-stopping-sensitive folds that amplify floating-point-level
  differences into different local optima) and with the seed: across seeds
  1/2/3/7/11, ``+DFA`` shaft-grouped recall on "earliest" ranges 0-4 of 5
  (mean 2.4) and "later_recruited" false positives range 2-21 (mean 11.8);
  *without* DFA the same sweep gives recall 1-4 of 5 (mean 3.4) and false
  positives 8-19 (mean 13.8). Those distributions overlap enough that this
  data cannot support a confident claim that DFA helps *or* hurts once
  shaft-based leakage is closed -- which is itself the finding: the
  plain-fold "win" was real for precision on the majority class (that part
  reproduces exactly, everywhere) but leakage-dependent for recall on the
  rare class, and once that leakage is closed there is not enough data left
  to tell whether recall is helped, hurt, or unaffected. See
  ``gnn_model_result/attention_deep_dfa_cv/`` and
  ``.../attention_deep_dfa_cv_shaft/`` (the saved seed-7 runs) and
  ``EVT_PROOF_small_sample_overfitting.md`` for the full multi-seed table.

None of these "fixes" manufacture the missing data (11-13 features, ~70
train nodes, 5 "earliest" positives total is still not enough for a
trustworthy neural net -- cross-validation, and grouped cross-validation
above all, is what actually surfaces how far that shortfall goes, rather
than a single favorable split or an unguarded fold assignment hiding it) --
they are worth comparing side by side with the baseline, not presented in
place of it.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import networkx as nx

from .data import load_graph_dataset, load_graph_kfold_datasets, GraphDataset
from .model import GNNConfig
from .train import CrossValidationResult, TrainingResult, cross_validate_gnn, train_gnn
from .visualize import plot_confusion_matrix, plot_cv_loss_curves, plot_loss_curves


def _describe(dataset: GraphDataset, result: TrainingResult) -> str:
    # The model reported below is whatever train_gnn ultimately returned: the restored
    # best-checkpoint weights when early stopping picked one, the last epoch's otherwise. Report
    # metrics from that *same* epoch's history entry, not history[-1], so the numbers printed here
    # always match the confusion matrices computed from this same model just below them.
    reported_epoch = result.best_epoch if result.best_epoch is not None else result.epochs
    i = reported_epoch - 1
    lines = [
        f"Graph: {len(dataset.node_names)} node(s) total, "
        f"{int(dataset.data.train_mask.sum() + dataset.data.val_mask.sum())} of them classifiable "
        f"channel node(s), {len(dataset.class_names)} role class(es): {dataset.class_names}.",
        f"Feature schema ({len(dataset.feature_names)} dims): {', '.join(dataset.feature_names)}.",
        f"Train/val node split: {int(dataset.data.train_mask.sum())} / {int(dataset.data.val_mask.sum())}.",
        "",
        result.model.describe(),
        "",
    ]
    if result.best_epoch is not None:
        lines.append(
            f"Best checkpoint (by {result.early_stopping_metric}): epoch {reported_epoch} of "
            f"{result.epochs} trained{' (stopped early)' if result.stopped_early else ''}: "
            f"loss={result.history['loss'][i]:.4f}, val_loss={result.history['val_loss'][i]:.4f}, "
            f"train_accuracy={result.history['train_accuracy'][i]:.3f}, "
            f"val_accuracy={result.history['val_accuracy'][i]:.3f}.")
    else:
        lines.append(
            f"Final epoch ({result.epochs}): loss={result.history['loss'][i]:.4f}, "
            f"val_loss={result.history['val_loss'][i]:.4f}, "
            f"train_accuracy={result.history['train_accuracy'][i]:.3f}, "
            f"val_accuracy={result.history['val_accuracy'][i]:.3f}.")
    lines += [
        "",
        f"Train confusion matrix (rows=true, cols=predicted, classes={result.class_names}):",
    ]
    for row_name, row in zip(result.class_names, result.train_confusion_matrix):
        lines.append(f"  {row_name:>15}: {list(int(v) for v in row)}")
    lines.append("")
    lines.append(f"Validation confusion matrix (rows=true, cols=predicted, classes={result.class_names}):")
    for row_name, row in zip(result.class_names, result.val_confusion_matrix):
        lines.append(f"  {row_name:>15}: {list(int(v) for v in row)}")
    return "\n".join(lines)


def run(graph_path: str | Path, output_dir: str | Path, target: str = "role", val_fraction: float = 0.3,
       hidden_channels: int = 16, num_layers: int = 2, dropout: float = 0.5, epochs: int = 200,
       lr: float = 0.01, weight_decay: float = 5e-4, seed: int = 7, drop_edge_p: float = 0.0,
       early_stopping_patience: int | None = None, early_stopping_metric: str = "val_loss",
       architecture: str = "gcn", heads: int = 4, concat_heads: bool = True,
       residual: bool = False, label_smoothing: float = 0.0) -> dict[str, object]:
    """Run the full GNN pipeline once on one already-built graph; returns the JSON summary it also writes.

    ``drop_edge_p``/``early_stopping_patience``/``architecture="gcn"`` all
    default to off/baseline, reproducing the unregularized GCN baseline
    exactly -- see this module's docstring.
    """
    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(
            f"{graph_path} does not exist -- build it first, e.g. "
            "`python -m object_model.run_object_model ...` (object_model_graph.graphml) or "
            "`extreme_event_agent.edf_workflow.build_seizure_graph` directly.")

    stem = graph_path.parent.name if graph_path.parent.name not in ("", ".") else graph_path.stem
    output_dir = Path(output_dir) / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = nx.read_graphml(graph_path)
    config = GNNConfig(hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout, seed=seed,
                       architecture=architecture, heads=heads, concat_heads=concat_heads, residual=residual)
    dataset = load_graph_dataset(graph, target=target, val_fraction=val_fraction, seed=seed)
    result = train_gnn(dataset, config=config, epochs=epochs, lr=lr, weight_decay=weight_decay,
                       drop_edge_p=drop_edge_p, early_stopping_patience=early_stopping_patience,
                       early_stopping_metric=early_stopping_metric, label_smoothing=label_smoothing)

    loss_curve_file = plot_loss_curves(result.history, output_dir / "gnn_loss_curve.png",
                                       title=f"{type(result.model).__name__} training")
    train_cm_file = plot_confusion_matrix(result.train_confusion_matrix, result.class_names,
                                          output_dir / "gnn_confusion_matrix_train.png",
                                          "Train confusion matrix")
    val_cm_file = plot_confusion_matrix(result.val_confusion_matrix, result.class_names,
                                        output_dir / "gnn_confusion_matrix_val.png",
                                        "Validation confusion matrix")

    summary_text = _describe(dataset, result)
    summary_file = output_dir / "gnn_model_summary.txt"
    summary_file.write_text(summary_text, encoding="utf-8")

    payload = {
        "graph_file": str(graph_path),
        "target": target,
        "class_names": dataset.class_names,
        "feature_names": dataset.feature_names,
        "num_nodes": len(dataset.node_names),
        "train_nodes": int(dataset.data.train_mask.sum()),
        "val_nodes": int(dataset.data.val_mask.sum()),
        "model_config": asdict(config),
        "trainable_parameters": result.model.num_parameters(),
        "drop_edge_p": drop_edge_p,
        "label_smoothing": label_smoothing,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_metric": result.early_stopping_metric,
        "stopped_early": result.stopped_early,
        "best_epoch": result.best_epoch,
        "epochs": result.epochs,
        "history": result.history,
        "train_confusion_matrix": result.train_confusion_matrix.tolist(),
        "val_confusion_matrix": result.val_confusion_matrix.tolist(),
        "train_classification_report": result.train_report,
        "val_classification_report": result.val_report,
        "figures": {
            "loss_curve": str(loss_curve_file),
            "train_confusion_matrix": str(train_cm_file),
            "val_confusion_matrix": str(val_cm_file),
        },
        "summary": summary_text,
        "summary_file": str(summary_file),
    }
    (output_dir / "gnn_model_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _describe_cv(result: CrossValidationResult) -> str:
    lines = [
        f"Cross-validated over {len(result.fold_results)} folds (StratifiedKFold) -- every "
        f"classifiable node was held out exactly once, across all folds.",
        "",
        result.fold_results[0].model.describe(),
        "",
        f"Mean over folds, at each fold's own reported checkpoint: "
        f"train_loss={result.mean_train_loss:.4f}, "
        f"val_loss={result.mean_val_loss:.4f} +/- {result.std_val_loss:.4f}, "
        f"val_accuracy={result.mean_val_accuracy:.3f} +/- {result.std_val_accuracy:.3f}, "
        f"val_macro_f1={result.mean_val_macro_f1:.3f} +/- {result.std_val_macro_f1:.3f}.",
        "",
        "Per-fold: " + "; ".join(
            f"fold {i + 1}: epoch {r.best_epoch or r.epochs}/{r.epochs}, "
            f"val_loss={r.history['val_loss'][(r.best_epoch or r.epochs) - 1]:.3f}"
            for i, r in enumerate(result.fold_results)),
        "",
        f"Out-of-fold confusion matrix (rows=true, cols=predicted, classes={result.class_names}, "
        "every classifiable node counted exactly once):",
    ]
    for row_name, row in zip(result.class_names, result.out_of_fold_confusion_matrix):
        lines.append(f"  {row_name:>15}: {list(int(v) for v in row)}")
    return "\n".join(lines)


def run_cross_validated(graph_path: str | Path, output_dir: str | Path, target: str = "role",
                        n_splits: int = 5, group_by_shaft: bool = False, hidden_channels: int = 16,
                        num_layers: int = 2, dropout: float = 0.5, epochs: int = 200, lr: float = 0.01,
                        weight_decay: float = 5e-4, seed: int = 7, drop_edge_p: float = 0.0,
                        early_stopping_patience: int | None = None, early_stopping_metric: str = "val_loss",
                        architecture: str = "gcn", heads: int = 4, concat_heads: bool = True,
                        residual: bool = False, label_smoothing: float = 0.0) -> dict[str, object]:
    """Like ``run``, but trains one model per ``StratifiedKFold`` (or, with ``group_by_shaft``,
    ``StratifiedGroupKFold``) fold instead of one model on one random split -- see
    ``gnn_model.data.load_graph_kfold_datasets`` and ``gnn_model.train.cross_validate_gnn`` for
    why: with only 5 "earliest" nodes total, a single 70/30 split's confusion matrix depends
    heavily on which 1-2 of them landed in validation. Same hyperparameters, same
    architecture/regularization knobs as ``run`` -- only how the train/val split is chosen (and
    how many times training happens) differs.

    ``group_by_shaft=True`` is the "smart folding": it keeps every contact on one electrode
    shaft together in the same fold, so a validation node's own neighbours from the same shaft
    (its strongest co-activation edges, physiologically the most similar channels in the whole
    graph) never sit in train -- see ``load_graph_kfold_datasets``'s docstring for the leakage
    this closes. It costs something real: shaft group sizes are uneven, so per-fold class
    balance is worse than plain ``StratifiedKFold`` (a fold can hold zero or several "earliest"
    nodes rather than ~1), which drags down mean per-fold ``val_macro_f1`` somewhat independent
    of model quality (see ``gnn_model_result/*_cv_shaft/``). The out-of-fold confusion matrix is
    still the reliable number either way.
    """
    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(
            f"{graph_path} does not exist -- build it first, e.g. "
            "`python -m object_model.run_object_model ...` (object_model_graph.graphml) or "
            "`extreme_event_agent.edf_workflow.build_seizure_graph` directly.")

    stem = graph_path.parent.name if graph_path.parent.name not in ("", ".") else graph_path.stem
    output_dir = Path(output_dir) / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = nx.read_graphml(graph_path)
    config = GNNConfig(hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout, seed=seed,
                       architecture=architecture, heads=heads, concat_heads=concat_heads, residual=residual)
    datasets = load_graph_kfold_datasets(graph, target=target, n_splits=n_splits, seed=seed,
                                         group_by_shaft=group_by_shaft)
    cv_result = cross_validate_gnn(datasets, config=config, epochs=epochs, lr=lr, weight_decay=weight_decay,
                                   drop_edge_p=drop_edge_p, early_stopping_patience=early_stopping_patience,
                                   early_stopping_metric=early_stopping_metric,
                                   label_smoothing=label_smoothing)

    loss_curve_file = plot_cv_loss_curves(
        [r.history for r in cv_result.fold_results], output_dir / "gnn_cv_loss_curve.png",
        title=f"{type(cv_result.fold_results[0].model).__name__} cross-validated training")
    oof_cm_file = plot_confusion_matrix(
        cv_result.out_of_fold_confusion_matrix, cv_result.class_names,
        output_dir / "gnn_cv_confusion_matrix_oof.png", f"Out-of-fold confusion matrix ({n_splits}-fold)")

    summary_text = _describe_cv(cv_result)
    summary_file = output_dir / "gnn_cv_summary.txt"
    summary_file.write_text(summary_text, encoding="utf-8")

    payload = {
        "graph_file": str(graph_path),
        "target": target,
        "n_splits": n_splits,
        "group_by_shaft": group_by_shaft,
        "class_names": cv_result.class_names,
        "model_config": asdict(config),
        "trainable_parameters": cv_result.fold_results[0].model.num_parameters(),
        "drop_edge_p": drop_edge_p,
        "label_smoothing": label_smoothing,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_metric": early_stopping_metric,
        "mean_train_loss": cv_result.mean_train_loss,
        "mean_val_loss": cv_result.mean_val_loss,
        "std_val_loss": cv_result.std_val_loss,
        "mean_val_accuracy": cv_result.mean_val_accuracy,
        "std_val_accuracy": cv_result.std_val_accuracy,
        "mean_val_macro_f1": cv_result.mean_val_macro_f1,
        "std_val_macro_f1": cv_result.std_val_macro_f1,
        "fold_epochs": [r.epochs for r in cv_result.fold_results],
        "fold_best_epochs": [r.best_epoch for r in cv_result.fold_results],
        "out_of_fold_confusion_matrix": cv_result.out_of_fold_confusion_matrix.tolist(),
        "out_of_fold_report": cv_result.out_of_fold_report,
        "figures": {
            "cv_loss_curve": str(loss_curve_file),
            "oof_confusion_matrix": str(oof_cm_file),
        },
        "summary": summary_text,
        "summary_file": str(summary_file),
    }
    (output_dir / "gnn_cv_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graph", default="object_model_result/sEEG-HFOs-8/object_model_graph.graphml",
                        help="GraphML file to train on (written by object_model.run_object_model or "
                             "extreme_event_agent.edf_workflow.build_seizure_graph)")
    parser.add_argument("--output", default="gnn_model_result")
    parser.add_argument("--target", default="role", help="Node attribute to classify")
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="L2 penalty on the weights. The main lever against val_loss "
                             "*diverging* rather than plateauing -- e.g. raising this from 5e-3 "
                             "to 0.2 on the attention_deep architecture turns a diverging val_loss "
                             "into one that plateaus at ~1.0 for 100+ epochs, same confusion "
                             "matrix -- see gnn_model.train's docstring")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="CrossEntropyLoss label smoothing (0 = off). Also caps how large "
                             "val_loss can get, but by capping model confidence rather than "
                             "weight magnitude -- on this graph it traded away confusion-matrix "
                             "quality at every value tried, unlike --weight-decay; see "
                             "gnn_model.train's docstring before reaching for this first")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--drop-edge-p", type=float, default=0.0,
                        help="Fraction of edges to randomly drop each training step (DropEdge "
                             "regularization; 0 = off, reproduces the unregularized baseline)")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Stop once --early-stopping-metric hasn't improved for this many "
                             "epochs, and report that best checkpoint instead of the final one "
                             "(unset = off)")
    parser.add_argument("--early-stopping-metric", choices=("val_loss", "val_accuracy", "val_macro_f1"),
                        default="val_loss",
                        help="What 'best' means for early stopping. 'val_loss' (default) is the "
                             "textbook choice; 'val_accuracy' is more robust to a heavily "
                             "class-weighted loss looking deceptively good in the first few epochs "
                             "before the model has learned anything, but on this imbalance a "
                             "regularized model can raise accuracy just by always predicting the "
                             "majority role; 'val_macro_f1' catches that -- see gnn_model.train")
    parser.add_argument("--architecture", choices=("gcn", "gat"), default="gcn",
                        help="'gcn' (default): SeizureGCN, fixed degree-normalized aggregation -- "
                             "the unregularized baseline. 'gat': SeizureGAT, learned per-neighbour "
                             "attention (GATv2) -- see gnn_model.model")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads; --architecture gat only")
    parser.add_argument("--no-concat-heads", dest="concat_heads", action="store_false",
                        help="Average attention heads in hidden layers instead of concatenating "
                             "them, so width stays hidden_channels regardless of heads -- keeps "
                             "a deep (--num-layers > 2) stack's parameter count from multiplying "
                             "by heads every layer. --architecture gat only")
    parser.add_argument("--residual", action="store_true",
                        help="Add GATv2Conv's own learnable skip connection to every layer -- "
                             "needed to keep a deep (--num-layers > 2) attention stack trainable "
                             "on a graph this small/low-diameter, where plain stacking oversmooths. "
                             "--architecture gat only")
    parser.add_argument("--cross-validate", type=int, default=None, metavar="N_SPLITS",
                        help="Train one model per StratifiedKFold fold (N_SPLITS folds) instead "
                             "of one model on one random --val-fraction split, and report the "
                             "out-of-fold confusion matrix (every classifiable node held out "
                             "exactly once) plus mean +/- std metrics across folds -- see "
                             "gnn_model.train.cross_validate_gnn. Unset (default) = off, the "
                             "single-split behaviour every other example in this module's "
                             "docstring uses. 5 matches this graph's 5 total 'earliest' nodes, "
                             "so each fold holds out almost exactly one of them.")
    parser.add_argument("--group-by-shaft", action="store_true",
                        help="With --cross-validate: use StratifiedGroupKFold grouped by "
                             "electrode shaft instead of plain StratifiedKFold, so contacts on "
                             "the same shaft (each other's strongest co-activation neighbours) "
                             "never split across train/val -- a harder, less flattering but "
                             "leakage-resistant fold assignment. Ignored without "
                             "--cross-validate. See gnn_model.data.load_graph_kfold_datasets.")
    args = parser.parse_args()

    common_kwargs = dict(
        target=args.target, hidden_channels=args.hidden_channels, num_layers=args.num_layers,
        dropout=args.dropout, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        seed=args.seed, drop_edge_p=args.drop_edge_p, early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric, concat_heads=args.concat_heads,
        residual=args.residual, label_smoothing=args.label_smoothing, architecture=args.architecture,
        heads=args.heads)

    if args.cross_validate is not None:
        payload = run_cross_validated(args.graph, args.output, n_splits=args.cross_validate,
                                      group_by_shaft=args.group_by_shaft, **common_kwargs)
    else:
        payload = run(args.graph, args.output, val_fraction=args.val_fraction, **common_kwargs)
    print(payload["summary"])
    graph_path = Path(args.graph)
    stem = graph_path.parent.name if graph_path.parent.name not in ("", ".") else graph_path.stem
    print(f"\nWrote GNN training outputs to {Path(args.output) / stem}")


if __name__ == "__main__":
    main()
