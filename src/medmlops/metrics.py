"""Evaluation metrics for segmentation and classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as_numpy(values) -> np.ndarray:
    import torch

    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def segmentation_metrics(pred_mask, target_mask, eps: float = 1e-7) -> dict[str, float]:
    """Pixel-level metrics averaged per image, with global aggregates.

    Args:
        pred_mask: array (B, H, W) or (B, 1, H, W), values in {0, 1}.
        target_mask: same shape as ``pred_mask``.
    """
    pred = _as_numpy(pred_mask).reshape(-1, pred_mask.shape[-2], pred_mask.shape[-1]) > 0.5
    target = _as_numpy(target_mask).reshape(-1, target_mask.shape[-2], target_mask.shape[-1]) > 0.5

    axes = tuple(range(1, pred.ndim))
    tp = (pred & target).sum(axis=axes).astype(np.float64)
    fp = (pred & ~target).sum(axis=axes).astype(np.float64)
    fn = (~pred & target).sum(axis=axes).astype(np.float64)
    tn = (~pred & ~target).sum(axis=axes).astype(np.float64)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    tp_sum, fp_sum, fn_sum = tp.sum(), fp.sum(), fn.sum()
    return {
        "dice": float(dice.mean()),
        "dice_std": float(dice.std()),
        "iou": float(iou.mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "specificity": float(specificity.mean()),
        "pixel_accuracy": float(accuracy.mean()),
        "global_dice": float((2 * tp_sum + eps) / (2 * tp_sum + fp_sum + fn_sum + eps)),
    }


def best_threshold(pred_prob, target_mask, low: float = 0.3, high: float = 0.75, steps: int = 19) -> float:
    prob = _as_numpy(pred_prob)
    target = _as_numpy(target_mask)
    best_t, best_dice = 0.5, -1.0
    for threshold in np.linspace(low, high, steps):
        score = segmentation_metrics(prob >= threshold, target)["dice"]
        if score > best_dice:
            best_t, best_dice = float(threshold), score
    return best_t


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        ece += in_bin.mean() * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def classification_metrics(y_true, y_prob, class_names: list[str] | None = None) -> dict:
    """Compute accuracy/AUC/F1/calibration and confusion matrix.

    Args:
        y_true: (N,) integer labels.
        y_prob: (N, C) probability matrix.
    """
    y_true = _as_numpy(y_true).astype(int).reshape(-1)
    y_prob = _as_numpy(y_prob)
    if y_prob.ndim == 1:
        y_prob = np.stack([1 - y_prob, y_prob], axis=1)
    n_classes = y_prob.shape[1]
    y_pred = y_prob.argmax(axis=1)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "ece": expected_calibration_error(y_prob, y_true),
    }

    try:
        if n_classes == 2:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["auc"] = float("nan")

    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=list(range(n_classes))))
    except ValueError:
        metrics["log_loss"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    metrics["confusion_matrix"] = cm.tolist()

    if n_classes == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = float(tp / max(tp + fn, 1))
        metrics["specificity"] = float(tn / max(tn + fp, 1))
        metrics["ppv"] = float(tp / max(tp + fp, 1))
        metrics["npv"] = float(tn / max(tn + fn, 1))

    if class_names:
        metrics["per_class_recall"] = {
            name: float(recall_score(y_true == i, y_pred == i, zero_division=0))
            for i, name in enumerate(class_names)
        }
    return metrics
