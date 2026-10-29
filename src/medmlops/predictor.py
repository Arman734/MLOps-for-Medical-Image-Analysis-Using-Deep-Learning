"""Inference service shared by the FastAPI app and the CLI."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .checkpoint import load_checkpoint
from .config import PROJECT_ROOT
from .data.transforms import load_image_for_inference
from .explain import gradcam_for_model, segmentation_tta
from .imaging import render_heatmap_overlay, render_mask_overlay, render_uncertainty, thumbnail
from .seed import describe_device, pick_device

DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"


@dataclass
class ModelBundle:
    task: str
    name: str
    path: Path
    model: torch.nn.Module
    meta: dict
    loaded_at: float
    mtime: float

    @property
    def threshold(self) -> float:
        return float(self.meta.get("threshold", 0.5))

    @property
    def image_size(self) -> int:
        return int(self.meta.get("image_size", 256))

    @property
    def class_names(self) -> list[str]:
        names = self.meta.get("class_names") or []
        return [str(n) for n in names]

    def public_info(self) -> dict[str, Any]:
        metrics = {
            k: round(float(v), 4)
            for k, v in (self.meta.get("metrics") or {}).items()
            if isinstance(v, (int, float)) and v == v
        }
        return {
            "id": self.name,
            "task": self.task,
            "arch": self.meta.get("arch"),
            "dataset": self.meta.get("dataset"),
            "image_size": self.image_size,
            "threshold": self.threshold,
            "class_names": self.class_names,
            "metrics": metrics,
            "run_id": self.meta.get("run_id"),
            "created_at": self.meta.get("created_at"),
            "git_commit": self.meta.get("git_commit"),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "loaded": True,
        }


class Predictor:
    def __init__(self, model_dir: str | Path | None = None, device: str = "auto") -> None:
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.device = pick_device(device)
        self.bundles: dict[str, ModelBundle] = {}
        self.load_errors: list[dict] = []

    # ------------------------------------------------------------------ load
    def refresh(self) -> None:
        self.bundles.clear()
        self.load_errors.clear()
        if not self.model_dir.exists():
            return
        for path in sorted(self.model_dir.glob("*.pt")):
            sidecar = path.with_suffix(".json")
            if not sidecar.exists():
                continue
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                task = meta.get("task", "segmentation")
                key = f"{task}:{path.stem}"
                if key in self.bundles:
                    continue
                model, _ = load_checkpoint(path, self.device)
                self.bundles[key] = ModelBundle(
                    task=task,
                    name=path.stem,
                    path=path,
                    model=model,
                    meta=meta,
                    loaded_at=time.time(),
                    mtime=path.stat().st_mtime,
                )
            except Exception as exc:  # pragma: no cover - corrupt checkpoint
                self.load_errors.append({"file": path.name, "error": str(exc)})

    def get(self, task: str, model_id: str | None = None) -> ModelBundle:
        candidates = [b for b in self.bundles.values() if b.task == task]
        if not candidates:
            raise LookupError(f"no {task} model available - train one first (see README)")
        if model_id:
            for bundle in candidates:
                if bundle.name == model_id or bundle.name == Path(model_id).stem:
                    return bundle
            raise LookupError(f"model '{model_id}' not found for task {task}")
        return max(candidates, key=lambda b: b.mtime)

    def info(self) -> dict[str, Any]:
        models = [b.public_info() for b in self.bundles.values()]
        return {
            "device": self.device.type,
            "device_name": describe_device(self.device),
            "model_dir": str(self.model_dir),
            "models": models,
            "segmentation_ready": any(b.task == "segmentation" for b in self.bundles.values()),
            "classification_ready": any(b.task == "classification" for b in self.bundles.values()),
            "load_errors": self.load_errors,
        }

    # ------------------------------------------------------------- inference
    def segment(self, image: Image.Image, model_id: str | None = None, use_tta: bool = True) -> dict:
        bundle = self.get("segmentation", model_id)
        started = time.perf_counter()
        tensor = load_image_for_inference(image, bundle.image_size, "segmentation").to(self.device)
        prob, uncertainty = segmentation_tta(bundle.model, tensor, use_tta=use_tta)
        latency_ms = (time.perf_counter() - started) * 1000.0

        mask = (prob >= bundle.threshold).astype(np.uint8)
        preview = thumbnail(image, 512)
        overlay = render_mask_overlay(
            np.asarray(Image.fromarray(preview).resize((bundle.image_size, bundle.image_size))),
            mask,
        )
        overlay = np.asarray(
            Image.fromarray(overlay).resize((preview.shape[1], preview.shape[0]), Image.Resampling.BILINEAR)
        )
        uncertainty_preview = render_uncertainty(
            np.asarray(Image.fromarray(preview).resize((bundle.image_size, bundle.image_size))),
            uncertainty,
        )
        uncertainty_preview = np.asarray(
            Image.fromarray(uncertainty_preview).resize((preview.shape[1], preview.shape[0]), Image.Resampling.BILINEAR)
        )
        lesion_fraction = float(mask.mean())
        inside = prob[mask > 0]
        return {
            "model_id": bundle.name,
            "model_version": bundle.meta.get("run_id"),
            "image_size": bundle.image_size,
            "threshold": bundle.threshold,
            "mask": mask,
            "probability": prob,
            "uncertainty": uncertainty,
            "preview": preview,
            "overlay": overlay,
            "uncertainty_overlay": uncertainty_preview,
            "lesion_fraction": lesion_fraction,
            "mean_probability": float(inside.mean()) if inside.size else 0.0,
            "uncertainty_mean": float(uncertainty.mean()),
            "latency_ms": latency_ms,
            "tta": bool(use_tta),
        }

    def classify(self, image: Image.Image, model_id: str | None = None, explain: bool = True) -> dict:
        bundle = self.get("classification", model_id)
        started = time.perf_counter()
        tensor = load_image_for_inference(image, bundle.image_size, "classification").to(self.device)
        with torch.no_grad():
            logits = bundle.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        class_index = int(probs.argmax())
        heatmap = gradcam_for_model(bundle.model, tensor, class_index) if explain else None
        latency_ms = (time.perf_counter() - started) * 1000.0

        preview = thumbnail(image, 512)
        overlay = None
        if heatmap is not None:
            overlay = render_heatmap_overlay(
                np.asarray(Image.fromarray(preview).resize((bundle.image_size, bundle.image_size))), heatmap
            )
            overlay = np.asarray(
                Image.fromarray(overlay).resize((preview.shape[1], preview.shape[0]), Image.Resampling.BILINEAR)
            )
        names = bundle.class_names or [str(i) for i in range(len(probs))]
        return {
            "model_id": bundle.name,
            "model_version": bundle.meta.get("run_id"),
            "image_size": bundle.image_size,
            "probabilities": [float(p) for p in probs],
            "labels": names,
            "predicted_index": class_index,
            "predicted_label": names[class_index] if class_index < len(names) else str(class_index),
            "confidence": float(probs[class_index]),
            "margin": float(probs[class_index] - np.sort(probs)[-2]) if len(probs) > 1 else 1.0,
            "preview": preview,
            "gradcam": overlay,
            "gradcam_map": heatmap,
            "latency_ms": latency_ms,
        }
