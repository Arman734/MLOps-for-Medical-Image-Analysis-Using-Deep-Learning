"""Consistent dark-themed figures logged as MLflow artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BG = "#0b1220"
PANEL = "#111a2e"
TEXT = "#dbe4f3"
MUTED = "#8b9bb8"
CYAN = "#22d3ee"
VIOLET = "#a78bfa"
ROSE = "#fb7185"
LIME = "#a3e635"


def _style(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1f2a44")
    ax.grid(color="#1f2a44", linewidth=0.8, alpha=0.7)


def _figure(figsize=(6.4, 4.0)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    _style(ax)
    return fig, ax


def _save(fig, path: str | Path, dpi: int = 130) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


def training_curves(history: dict[str, list[float]], path: str | Path, metric_name: str) -> Path:
    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=BG)
    for ax in axes:
        _style(ax)

    axes[0].plot(epochs, history["train_loss"], color=CYAN, linewidth=2, label="train")
    axes[0].plot(epochs, history["val_loss"], color=VIOLET, linewidth=2, label="validation")
    axes[0].set_title("Loss", color=TEXT, fontsize=12, loc="left")
    axes[0].set_xlabel("epoch", color=MUTED)
    axes[0].legend(facecolor=PANEL, edgecolor="#1f2a44", labelcolor=TEXT, fontsize=9)

    metric_keys = [k for k in history if k.startswith("val_") and k != "val_loss"]
    palette = [LIME, ROSE, CYAN]
    for key, color in zip(metric_keys, palette, strict=False):
        axes[1].plot(epochs, history[key], color=color, linewidth=2, label=key.replace("val_", ""))
    axes[1].set_title(f"Validation {metric_name}", color=TEXT, fontsize=12, loc="left")
    axes[1].set_xlabel("epoch", color=MUTED)
    axes[1].legend(facecolor=PANEL, edgecolor="#1f2a44", labelcolor=TEXT, fontsize=9)

    fig.tight_layout()
    return _save(fig, path)


def seg_prediction_grid(
    images: np.ndarray,
    targets: np.ndarray,
    probs: np.ndarray,
    path: str | Path,
    threshold: float = 0.5,
    n: int = 4,
) -> Path:
    n = min(n, len(images))
    fig, axes = plt.subplots(3, n, figsize=(3.0 * n, 8.4), facecolor=BG)
    if n == 1:
        axes = axes.reshape(3, 1)
    for col in range(n):
        image = np.clip(images[col].transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        target = targets[col].squeeze()
        prob = probs[col].squeeze()
        pred = (prob >= threshold).astype(np.float32)

        axes[0, col].imshow(image)
        axes[0, col].set_title(f"case {col + 1}", color=TEXT, fontsize=10, loc="left")
        axes[1, col].imshow(image)
        axes[1, col].imshow(np.ma.masked_where(target < 0.5, target), cmap="spring", alpha=0.55, vmin=0, vmax=1)
        axes[2, col].imshow(image)
        axes[2, col].imshow(np.ma.masked_where(pred < 0.5, pred), cmap="winter", alpha=0.55, vmin=0, vmax=1)
        axes[2, col].contour(prob, levels=[threshold], colors=[CYAN], linewidths=1.0)
        for row, label in enumerate(("image", "ground truth", f"prediction (t={threshold:.2f})")):
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].text(0.02, 0.94, label, transform=axes[row, col].transAxes, color=TEXT, fontsize=9, va="top")
    fig.tight_layout()
    return _save(fig, path, dpi=120)


def confusion_matrix_plot(cm: list[list[int]], class_names: list[str], path: str | Path) -> Path:
    matrix = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 4.3), facecolor=BG)
    _style(ax)
    ax.grid(False)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(normalized, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, color=MUTED, fontsize=9)
    ax.set_yticklabels(class_names, color=MUTED, fontsize=9)
    ax.set_xlabel("predicted", color=MUTED)
    ax.set_ylabel("actual", color=MUTED)
    ax.set_title("Confusion matrix (row-normalised)", color=TEXT, fontsize=11, loc="left")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{int(matrix[i, j])}\n{normalized[i, j]:.0%}",
                ha="center",
                va="center",
                color="#0b1220" if normalized[i, j] > 0.55 else TEXT,
                fontsize=9,
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=MUTED)
    cbar.outline.set_edgecolor("#1f2a44")
    fig.tight_layout()
    return _save(fig, path)


def roc_curve_plot(y_true: np.ndarray, probs: np.ndarray, class_names: list[str], path: str | Path) -> Path:
    from sklearn.metrics import roc_curve

    fig, ax = _figure((5.4, 4.4))
    if probs.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        ax.plot(fpr, tpr, color=CYAN, linewidth=2, label="ROC")
        ax.fill_between(fpr, tpr, color=CYAN, alpha=0.08)
    else:
        from sklearn.metrics import roc_curve as _roc

        for index, name in enumerate(class_names):
            fpr, tpr, _ = _roc((y_true == index).astype(int), probs[:, index])
            ax.plot(fpr, tpr, linewidth=1.6, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1)
    ax.set_xlabel("false positive rate", color=MUTED)
    ax.set_ylabel("true positive rate", color=MUTED)
    ax.set_title("ROC curve", color=TEXT, fontsize=12, loc="left")
    ax.legend(facecolor=PANEL, edgecolor="#1f2a44", labelcolor=TEXT, fontsize=9, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def probability_distribution(y_true: np.ndarray, probs: np.ndarray, class_names: list[str], path: str | Path) -> Path:
    fig, ax = _figure((6.4, 4.0))
    positive = probs[:, 1] if probs.shape[1] == 2 else probs.max(axis=1)
    for label, color in zip(sorted(set(y_true.tolist())), (CYAN, ROSE, VIOLET, LIME), strict=False):
        subset = positive[y_true == label]
        ax.hist(subset, bins=20, range=(0, 1), alpha=0.65, color=color, label=class_names[label] if label < len(class_names) else str(label))
    ax.set_xlabel("predicted probability", color=MUTED)
    ax.set_ylabel("count", color=MUTED)
    ax.set_title("Confidence distribution", color=TEXT, fontsize=12, loc="left")
    ax.legend(facecolor=PANEL, edgecolor="#1f2a44", labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    return _save(fig, path)
