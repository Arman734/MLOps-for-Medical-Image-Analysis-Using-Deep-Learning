"""Inference endpoints: segmentation, diagnosis, explainability and feedback."""

from __future__ import annotations

import io

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from medmlops.data.common import compute_image_features
from medmlops.imaging import png_b64

from ..deps import get_state
from ..metrics import PREDICTION_COUNT, PREDICTION_LATENCY
from ..services import AppState

router = APIRouter(prefix="/api", tags=["inference"])


class FeedbackRequest(BaseModel):
    prediction_id: str
    verdict: str
    comment: str = ""


def _feature_image(image: Image.Image, task: str, size: int) -> np.ndarray:
    """Match the data-prep resize so drift features share the reference scale."""
    resample = Image.Resampling.BICUBIC if task == "segmentation" else Image.Resampling.BILINEAR
    return np.asarray(image.resize((size, size), resample))


async def _read_upload(file: UploadFile, state: AppState) -> Image.Image:
    data = await file.read()
    limit = state.settings.max_upload_mb * 1024 * 1024
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"image exceeds {state.settings.max_upload_mb} MB limit")
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"unsupported image file: {exc}") from exc
    image = image.convert("RGB")
    if min(image.size) < 32:
        raise HTTPException(status_code=400, detail="image must be at least 32x32 pixels")
    return image


@router.post("/predict/segmentation")
async def predict_segmentation(
    file: UploadFile = File(...),
    model_id: str | None = Form(None),
    tta: bool = Form(True),
    state: AppState = Depends(get_state),
) -> dict:
    image = await _read_upload(file, state)
    try:
        result = state.predictor.segment(image, model_id=model_id, use_tta=tta)
    except LookupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    features = compute_image_features(_feature_image(image, "segmentation", result["image_size"]))
    label = "lesion" if result["lesion_fraction"] > 0.005 else "no_lesion"
    prediction_id = state.store.log_prediction(
        task="segmentation",
        model_id=result["model_id"],
        model_version=result["model_version"],
        filename=file.filename,
        latency_ms=result["latency_ms"],
        label=label,
        confidence=result["mean_probability"],
        features=features,
        meta={"lesion_fraction": round(result["lesion_fraction"], 5), "tta": result["tta"], "threshold": result["threshold"]},
    )
    PREDICTION_COUNT.labels("segmentation", label).inc()
    PREDICTION_LATENCY.labels("segmentation").observe(result["latency_ms"] / 1000.0)

    preview = result["preview"]
    return {
        "prediction_id": prediction_id,
        "task": "segmentation",
        "model_id": result["model_id"],
        "model_version": result["model_version"],
        "threshold": result["threshold"],
        "tta": result["tta"],
        "latency_ms": round(result["latency_ms"], 2),
        "lesion_fraction": round(result["lesion_fraction"], 5),
        "mean_probability": round(result["mean_probability"], 4),
        "uncertainty_mean": round(result["uncertainty_mean"], 4),
        "width": int(preview.shape[1]),
        "height": int(preview.shape[0]),
        "preview_base64": png_b64(preview),
        "overlay_base64": png_b64(result["overlay"]),
        "mask_base64": png_b64((result["mask"] * 255).astype(np.uint8)),
        "uncertainty_base64": png_b64(result["uncertainty_overlay"]),
        "features": {k: round(v, 4) for k, v in features.items()},
    }


@router.post("/predict/classification")
async def predict_classification(
    file: UploadFile = File(...),
    model_id: str | None = Form(None),
    explain: bool = Form(True),
    state: AppState = Depends(get_state),
) -> dict:
    image = await _read_upload(file, state)
    try:
        result = state.predictor.classify(image, model_id=model_id, explain=explain)
    except LookupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    features = compute_image_features(_feature_image(image, "classification", result["image_size"]))
    prediction_id = state.store.log_prediction(
        task="classification",
        model_id=result["model_id"],
        model_version=result["model_version"],
        filename=file.filename,
        latency_ms=result["latency_ms"],
        label=result["predicted_label"],
        confidence=result["confidence"],
        features=features,
        meta={"margin": round(result["margin"], 4), "probabilities": result["probabilities"]},
    )
    PREDICTION_COUNT.labels("classification", result["predicted_label"]).inc()
    PREDICTION_LATENCY.labels("classification").observe(result["latency_ms"] / 1000.0)

    preview = result["preview"]
    return {
        "prediction_id": prediction_id,
        "task": "classification",
        "model_id": result["model_id"],
        "model_version": result["model_version"],
        "latency_ms": round(result["latency_ms"], 2),
        "predicted_label": result["predicted_label"],
        "predicted_index": result["predicted_index"],
        "confidence": round(result["confidence"], 4),
        "margin": round(result["margin"], 4),
        "labels": result["labels"],
        "probabilities": [round(p, 4) for p in result["probabilities"]],
        "width": int(preview.shape[1]),
        "height": int(preview.shape[0]),
        "preview_base64": png_b64(preview),
        "gradcam_base64": png_b64(result["gradcam"]) if result["gradcam"] is not None else None,
        "features": {k: round(v, 4) for k, v in features.items()},
    }


@router.post("/feedback")
def submit_feedback(payload: FeedbackRequest, state: AppState = Depends(get_state)) -> dict:
    if payload.verdict not in {"correct", "incorrect", "uncertain"}:
        raise HTTPException(status_code=400, detail="verdict must be correct, incorrect or uncertain")
    state.store.add_feedback(payload.prediction_id, payload.verdict, payload.comment)
    return {"status": "recorded", "prediction_id": payload.prediction_id, "verdict": payload.verdict}


@router.get("/predictions/recent")
def recent_predictions(limit: int = 25, task: str | None = None, state: AppState = Depends(get_state)) -> dict:
    return {"predictions": state.store.recent(limit=max(1, min(limit, 200)), task=task)}
