"""Figures for a ``gnn_model`` training run: the loss curves and the confusion matrices.

Follows the same headless-backend convention as ``model.visualize``/
``object_model.figure``: import and set ``Agg`` inside each function, so
importing this module never requires a display.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["plot_loss_curves", "plot_confusion_matrix", "plot_cv_loss_curves"]


def plot_loss_curves(history: dict[str, list[float]], output: str | Path,
                     title: str = "SeizureGCN training") -> Path:
    """``loss`` (train) vs ``val_loss`` per epoch, the standard training-curve diagnostic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    epochs = range(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["loss"], label="loss (train)", color="#1f77b4")
    ax.plot(epochs, history["val_loss"], label="val_loss", color="#d62728")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_cv_loss_curves(fold_histories: list[dict[str, list[float]]], output: str | Path,
                        title: str = "Cross-validated training") -> Path:
    """One thin line per fold (faint, full length) plus the mean curve (bold), for both ``loss``
    and ``val_loss`` -- shows whether a training trajectory is a shared pattern across folds or an
    artefact of whichever nodes one particular fold happened to hold out. Folds may have
    different lengths (independent early stopping per fold); the mean line stops at the
    *shortest* fold's length rather than being computed over however many folds are still
    running at each epoch -- extending it past that would change which folds contribute partway
    through, which can make the mean jump discontinuously right when the shortest fold's own
    (possibly atypical) curve simply stops being counted, not because anything about training
    changed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    fig, ax = plt.subplots(figsize=(8, 5))
    min_len = min(len(h["loss"]) for h in fold_histories)
    for key, color, label in (("loss", "#1f77b4", "loss (train)"), ("val_loss", "#d62728", "val_loss")):
        for fold_index, history in enumerate(fold_histories):
            epochs = range(1, len(history[key]) + 1)
            ax.plot(epochs, history[key], color=color, alpha=0.25, linewidth=1,
                    label=f"{label}, per fold" if fold_index == 0 else None)
        mean_curve = [np.mean([h[key][e] for h in fold_histories]) for e in range(min_len)]
        ax.plot(range(1, min_len + 1), mean_curve, color=color, linewidth=2.5, label=f"{label} (mean)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output: str | Path, title: str) -> Path:
    """Annotated confusion-matrix heatmap (rows = true role, columns = predicted role)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    side = max(4, 2 + len(class_names))
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted role")
    ax.set_ylabel("true role")
    ax.set_title(title)
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
