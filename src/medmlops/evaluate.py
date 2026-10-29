"""Standalone evaluation of a saved checkpoint on a held-out split."""

from __future__ import annotations

import json
from pathlib import Path

from .checkpoint import load_checkpoint
from .config import PROJECT_ROOT, load_config
from .data import isic, medmnist
from .metrics import classification_metrics
from .seed import describe_device, pick_device


def evaluate_checkpoint(
    config_path: str,
    checkpoint: str | Path | None = None,
    split: str = "test",
    device: str = "auto",
    write: bool = True,
) -> dict:
    cfg = load_config(config_path)
    torch_device = pick_device(device)
    ckpt = Path(checkpoint) if checkpoint else PROJECT_ROOT / cfg.output.model_dir / cfg.output.name
    model, meta = load_checkpoint(ckpt, torch_device)

    from .train import DiceBCELoss, _loader, evaluate_segmentation, predict_probabilities

    if meta["task"] == "segmentation":
        dataset = isic.load_split(cfg, split, train=False)
        loader = _loader(dataset, cfg, train=False, device=torch_device)
        threshold = float(meta.get("threshold", 0.5))
        metrics, _, probs, targets = evaluate_segmentation(
            model, loader, DiceBCELoss(), torch_device, threshold=threshold, collect_probs=True
        )
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics["threshold"] = threshold
        metrics["samples"] = int(len(targets))
    else:
        dataset = medmnist.load_split(cfg, split, train=False)
        loader = _loader(dataset, cfg, train=False, device=torch_device)
        probs, labels = predict_probabilities(model, loader, torch_device)
        metrics = classification_metrics(labels, probs, meta.get("class_names"))
        metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        metrics["samples"] = int(len(labels))

    metrics["checkpoint"] = str(ckpt)
    metrics["split"] = split
    metrics["device"] = describe_device(torch_device)

    if write:
        out = PROJECT_ROOT / "metrics" / f"eval_{cfg.experiment}_{split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metrics["output"] = str(out)
    return metrics
