"""System, model registry and demo-case endpoints."""

from __future__ import annotations

import base64
import io
import platform
import sys

from fastapi import APIRouter, Depends, HTTPException
from PIL import Image
from pydantic import BaseModel

from medmlops import __version__, tracking

from ..deps import get_state
from ..metrics import refresh_model_gauges
from ..services import AppState, find_reference_stats

router = APIRouter(prefix="/api", tags=["system"])


class PromoteRequest(BaseModel):
    registry_name: str
    run_id: str


@router.get("/health")
def health(state: AppState = Depends(get_state)) -> dict:
    info = state.predictor.info()
    return {
        "status": "ok",
        "version": __version__,
        "uptime_s": round(state.uptime_s, 1),
        "device": info["device"],
        "device_name": info["device_name"],
        "segmentation_ready": info["segmentation_ready"],
        "classification_ready": info["classification_ready"],
    }


@router.get("/system")
def system(state: AppState = Depends(get_state)) -> dict:
    import fastapi
    import torch

    info = state.predictor.info()
    try:
        track = tracking.tracking_info()
    except Exception as exc:  # pragma: no cover - mlflow always installed
        track = {"backend": "unavailable", "error": str(exc)}
    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fastapi": fastapi.__version__,
        "device": info["device"],
        "device_name": info["device_name"],
        "models_loaded": len(info["models"]),
        "tracking": track,
        "datasets": state.dataset_summaries(),
        "uptime_s": round(state.uptime_s, 1),
    }


@router.get("/models")
def models(state: AppState = Depends(get_state)) -> dict:
    info = state.predictor.info()
    registry: list[dict] = []
    try:
        registry = tracking.list_models()
    except Exception:  # pragma: no cover - registry store may be empty
        registry = []
    return {**info, "registry": registry}


@router.post("/models/refresh")
def refresh_models(state: AppState = Depends(get_state)) -> dict:
    state.predictor.refresh()
    state.references.update(find_reference_stats())
    refresh_model_gauges(state.predictor)
    return state.predictor.info()


@router.post("/models/promote")
def promote_model(payload: PromoteRequest) -> dict:
    try:
        return tracking.promote_by_run(payload.registry_name, payload.run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/models/champion/{registry_name}")
def champion(registry_name: str) -> dict:
    import mlflow

    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(registry_name, "champion")
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"no champion for '{registry_name}'") from exc
    run = client.get_run(version.run_id)
    return {
        "name": registry_name,
        "version": version.version,
        "run_id": version.run_id,
        "metrics": run.data.metrics,
        "description": version.description,
    }


@router.get("/demo/{task}")
def demo_case(task: str, state: AppState = Depends(get_state)) -> dict:
    """Return a real held-out test image so the console can be tried instantly."""
    if task not in {"segmentation", "classification"}:
        raise HTTPException(status_code=400, detail="task must be segmentation or classification")
    from medmlops.config import PROJECT_ROOT

    try:
        if task == "segmentation":
            import numpy as np

            path = PROJECT_ROOT / "data" / "processed" / "isic2018" / "test.npz"
            if not path.exists():
                raise FileNotFoundError("ISIC test split not found")
            bundle = np.load(path)
            index = int(len(bundle["images"]) * 0.37) % len(bundle["images"])
            image = Image.fromarray(bundle["images"][index])
            name = str(bundle["names"][index])
        else:
            from medmlops.config import load_config
            from medmlops.data import medmnist

            cfg = load_config("configs/pneumonia_densenet.yaml")
            base = medmnist._base_dataset(cfg, "test")
            index = int(len(base) * 0.61) % len(base)
            image, _ = base[index]
            image = image.convert("RGB") if image.mode != "RGB" else image
            name = f"{cfg.data.name}_test_{index:04d}"
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"demo case unavailable: {exc}") from exc

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        "task": task,
        "name": name,
        "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "width": image.width,
        "height": image.height,
    }
