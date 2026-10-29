"""Checkpoint format shared by training, evaluation, the API and the CLI."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


def build_model_from_meta(meta: dict, device: str | torch.device = "cpu") -> torch.nn.Module:
    from .models import UNet, build_densenet

    arch = meta["arch"]
    kwargs = dict(meta.get("model_kwargs", {}))
    if arch == "unet":
        model = UNet(**kwargs)
    elif arch == "densenet121":
        model = build_densenet(pretrained=False, **kwargs)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported checkpoint architecture: {arch}")
    model.to(device)
    model.eval()
    return model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    meta: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "meta": meta}
    torch.save(payload, path)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    meta = payload["meta"]
    model = build_model_from_meta(meta, device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, meta


def new_meta(*, task: str, arch: str, model_kwargs: dict, image_size: int, dataset: str, run_id: str,
             git_commit: str, config_path: str, **extra) -> dict:
    return {
        "task": task,
        "arch": arch,
        "model_kwargs": model_kwargs,
        "image_size": int(image_size),
        "dataset": dataset,
        "run_id": run_id,
        "git_commit": git_commit,
        "config_path": config_path,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }
