"""Figures for a ``gnn_model`` training run: the loss curves and the confusion matrices.

Follows the same headless-backend convention as ``model.visualize``/
``object_model.figure``: import and set ``Agg`` inside each function, so
importing this module never requires a display.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["plot_loss_curves", "plot_confusion_matrix"]


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
