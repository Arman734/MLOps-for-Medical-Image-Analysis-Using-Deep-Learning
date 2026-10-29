"""Model card generation for every trained artifact."""

from __future__ import annotations

from datetime import UTC, datetime


def _metric_rows(metrics: dict) -> str:
    keys = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "pixel_accuracy",
        "accuracy",
        "auc",
        "f1_macro",
        "sensitivity",
        "specificity",
        "ece",
        "log_loss",
    ]
    seen: set[str] = set()
    rows = []
    for key in keys:
        if key in metrics and key not in seen:
            seen.add(key)
            value = metrics[key]
            if isinstance(value, (int, float)):
                rows.append(f"| {key} | {value:.4f} |")
    return "\n".join(rows) if rows else "| - | - |"


def model_card(
    *,
    title: str,
    task: str,
    dataset: str,
    dataset_size: dict,
    model_desc: str,
    metrics: dict,
    threshold: float | None,
    run_id: str,
    git_commit: str,
    device: str,
    extra: str = "",
) -> str:
    split_line = ", ".join(f"{k}: {v}" for k, v in dataset_size.items())
    threshold_line = f"\n- **Operating threshold:** {threshold:.2f}" if threshold is not None else ""
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""# Model Card — {title}

> Generated automatically by the MedMLOps training pipeline on {generated}.

## Overview
- **Task:** {task}
- **Architecture:** {model_desc}
- **Dataset:** {dataset} ({split_line})
- **MLflow run:** `{run_id}`
- **Git commit:** `{git_commit}`
- **Hardware:** {device}

## Intended use
Research and education: benchmarking reproducible medical-imaging pipelines
(data preparation → training → tracking → registry → deployment → monitoring).
The system surfaces model outputs as *assistive* signals for a clinician.

## Out of scope / limitations
- **Not a medical device.** No regulatory clearance (CE/FDA); must not be used for diagnosis or treatment decisions.
- Evaluated on a single public dataset; performance on other scanners, populations
  or acquisition protocols is unknown. Lesion-type and skin-tone coverage is limited.
- Segmentation masks inherit the annotation noise of the source dataset.
- Predictions can be confidently wrong; always review with a qualified clinician.

## Evaluation (held-out test split)
| metric | value |
| --- | --- |
{_metric_rows(metrics)}
{threshold_line}
{extra}
## Reproducibility
```bash
dvc repro          # rebuild data + model from the versioned pipeline
mlflow ui          # inspect this run (tracking URI in .env)
```
Every run records hyperparameters, dataset hashes, metrics and artifacts.
"""


def segmentation_card(**kwargs) -> str:
    kwargs.setdefault("title", "ISIC 2018 lesion segmentation (U-Net)")
    kwargs.setdefault("task", "Binary lesion segmentation (dermatology)")
    return model_card(**kwargs)


def classification_card(**kwargs) -> str:
    kwargs.setdefault("title", "PneumoniaMNIST diagnosis (DenseNet-121)")
    kwargs.setdefault("task", "Binary chest X-ray classification (normal vs pneumonia)")
    return model_card(**kwargs)
