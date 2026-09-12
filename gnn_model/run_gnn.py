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

None of these "fixes" manufacture the missing data (11 features, ~70 train
nodes, 5 "earliest" positives total is still not enough for a trustworthy
neural net -- ``attention/`` still only catches 1 of the 2 "earliest" val
nodes, same as the reckless baseline) -- they are worth comparing side by
side with the baseline, not presented in place of it.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import networkx as nx

from .data import load_graph_dataset, GraphDataset
from .model import GNNConfig
from .train import TrainingResult, train_gnn
from .visualize import plot_confusion_matrix, plot_loss_curves


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
       architecture: str = "gcn", heads: int = 4) -> dict[str, object]:
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
                       architecture=architecture, heads=heads)
    dataset = load_graph_dataset(graph, target=target, val_fraction=val_fraction, seed=seed)
    result = train_gnn(dataset, config=config, epochs=epochs, lr=lr, weight_decay=weight_decay,
                       drop_edge_p=drop_edge_p, early_stopping_patience=early_stopping_patience,
                       early_stopping_metric=early_stopping_metric)

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
    parser.add_argument("--weight-decay", type=float, default=5e-4)
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
    args = parser.parse_args()

    payload = run(args.graph, args.output, target=args.target, val_fraction=args.val_fraction,
                 hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout,
                 epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
                 drop_edge_p=args.drop_edge_p, early_stopping_patience=args.early_stopping_patience,
                 early_stopping_metric=args.early_stopping_metric,
                 architecture=args.architecture, heads=args.heads)
    print(payload["summary"])
    graph_path = Path(args.graph)
    stem = graph_path.parent.name if graph_path.parent.name not in ("", ".") else graph_path.stem
    print(f"\nWrote GNN training outputs to {Path(args.output) / stem}")


if __name__ == "__main__":
    main()
