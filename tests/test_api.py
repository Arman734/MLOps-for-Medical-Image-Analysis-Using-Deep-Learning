"""FastAPI contract tests with tiny real checkpoints (CPU, no dataset needed)."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.app import create_app
from api.services import AppState, Settings
from medmlops.checkpoint import new_meta, save_checkpoint
from medmlops.data.common import build_reference_stats
from medmlops.models import UNet, build_densenet
from medmlops.monitor import MonitoringStore
from medmlops.predictor import Predictor


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_models(model_dir):
    unet = UNet(base_channels=4, depth=2, dropout=0.0)
    save_checkpoint(
        model_dir / "unit_unet.pt",
        unet,
        new_meta(
            task="segmentation",
            arch="unet",
            model_kwargs={"base_channels": 4, "depth": 2},
            image_size=64,
            dataset="unit",
            run_id="run-seg",
            git_commit="abc1234",
            config_path="configs/isic_unet.yaml",
            threshold=0.5,
            metrics={"dice": 0.81},
        ),
    )
    densenet = build_densenet(num_classes=2, pretrained=False)
    save_checkpoint(
        model_dir / "unit_densenet.pt",
        densenet,
        new_meta(
            task="classification",
            arch="densenet121",
            model_kwargs={"num_classes": 2},
            image_size=64,
            dataset="unit",
            run_id="run-clf",
            git_commit="abc1234",
            config_path="configs/pneumonia_densenet.yaml",
            class_names=["normal", "pneumonia"],
            metrics={"auc": 0.9},
        ),
    )


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    model_dir = root / "models"
    model_dir.mkdir()
    _build_models(model_dir)
    predictor = Predictor(model_dir, device="cpu")
    predictor.refresh()
    assert len(predictor.bundles) == 2

    rng = np.random.default_rng(3)
    images = [rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8) for _ in range(24)]
    reference = build_reference_stats(images, dataset="unit", sample_limit=24)
    state = AppState(
        settings=Settings(
            model_dir=model_dir,
            db_path=root / "monitor.db",
            cors_origins=["http://localhost:5173"],
            max_upload_mb=5,
            device="cpu",
        ),
        predictor=predictor,
        store=MonitoringStore(root / "monitor.db"),
        references={"segmentation": reference, "classification": reference},
    )
    application = create_app()
    application.state.services = state
    with TestClient(application) as client:
        yield client


@pytest.fixture()
def image_bytes(rng) -> bytes:
    return _png(Image.fromarray(rng.integers(0, 255, size=(90, 70, 3), dtype=np.uint8)))


def test_health_and_system(app):
    health = app.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["segmentation_ready"] is True
    assert health["classification_ready"] is True

    system = app.get("/api/system").json()
    for key in ("python", "torch", "device", "tracking", "datasets", "uptime_s"):
        assert key in system
    assert system["models_loaded"] == 2
    assert "segmentation" in system["datasets"]


def test_models_endpoint(app):
    payload = app.get("/api/models").json()
    assert payload["segmentation_ready"] and payload["classification_ready"]
    names = {model["id"] for model in payload["models"]}
    assert names == {"unit_unet", "unit_densenet"}
    assert "registry" in payload

    refreshed = app.post("/api/models/refresh").json()
    assert len(refreshed["models"]) == 2


def test_prometheus_endpoint(app):
    response = app.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "medmlops_api_requests_total" in response.text
    assert "medmlops_model_ready" in response.text


def test_segmentation_prediction_and_ledger(app, image_bytes):
    response = app.post(
        "/api/predict/segmentation",
        files={"file": ("case.png", image_bytes, "image/png")},
        data={"tta": "false"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task"] == "segmentation"
    assert body["model_id"] == "unit_unet"
    assert 0.0 <= body["lesion_fraction"] <= 1.0
    assert body["width"] > 0 and body["height"] > 0
    assert base64.b64decode(body["overlay_base64"])[:4] == b"\x89PNG"
    assert base64.b64decode(body["mask_base64"])[:4] == b"\x89PNG"
    assert set(body["features"]) == {"brightness", "contrast", "entropy", "saturation", "colorfulness"}

    recent = app.get("/api/predictions/recent", params={"task": "segmentation"}).json()
    assert recent["predictions"][0]["id"] == body["prediction_id"]


def test_classification_prediction_with_gradcam(app, image_bytes):
    response = app.post(
        "/api/predict/classification",
        files={"file": ("cxr.png", image_bytes, "image/png")},
        data={"explain": "true"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task"] == "classification"
    assert body["labels"] == ["normal", "pneumonia"]
    assert len(body["probabilities"]) == 2
    assert sum(body["probabilities"]) == pytest.approx(1.0, abs=1e-3)
    assert body["predicted_label"] in body["labels"]
    assert body["gradcam_base64"]
    assert base64.b64decode(body["gradcam_base64"])[:4] == b"\x89PNG"


def test_feedback_flow(app, image_bytes):
    prediction = app.post(
        "/api/predict/classification",
        files={"file": ("cxr.png", image_bytes, "image/png")},
        data={"explain": "false"},
    ).json()

    ok = app.post(
        "/api/feedback",
        json={"prediction_id": prediction["prediction_id"], "verdict": "correct", "comment": "agree"},
    )
    assert ok.status_code == 200 and ok.json()["status"] == "recorded"

    bad = app.post(
        "/api/feedback", json={"prediction_id": prediction["prediction_id"], "verdict": "maybe"}
    )
    assert bad.status_code == 400


def test_monitoring_endpoints(app, image_bytes):
    app.post("/api/predict/segmentation", files={"file": ("case.png", image_bytes, "image/png")})
    summary = app.get("/api/monitoring/summary", params={"window": 100}).json()
    assert summary["total_predictions"] >= 1
    assert "drift" in summary and "labels" in summary

    drift = app.get("/api/monitoring/drift", params={"task": "segmentation", "window": 100}).json()
    assert drift["status"] in {"stable", "watch", "alert", "insufficient_data", "no_reference"}

    export = app.get("/api/monitoring/export.csv")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("text/csv")
    assert export.text.splitlines()[0].startswith("id,timestamp,task")


def test_pipeline_endpoints(app):
    configs = app.get("/api/pipeline/configs").json()["configs"]
    assert "configs/isic_unet.yaml" in configs

    jobs = app.get("/api/pipeline/jobs").json()["jobs"]
    assert isinstance(jobs, list)

    missing = app.get("/api/pipeline/jobs/does-not-exist")
    assert missing.status_code == 404

    rejected = app.post("/api/pipeline/train", json={"config": "/etc/passwd"})
    assert rejected.status_code == 400


def test_experiments_and_registry(app):
    experiments = app.get("/api/experiments")
    assert experiments.status_code == 200
    assert "tracking" in experiments.json()

    registry = app.get("/api/registry")
    assert registry.status_code == 200
    assert "models" in registry.json()

    unknown_run = app.get("/api/experiments/nonexistent-run")
    assert unknown_run.status_code == 404

    promote = app.post(
        "/api/models/promote",
        json={"registry_name": "does-not-exist", "run_id": "nope"},
    )
    assert promote.status_code == 404


def test_upload_validation(app):
    bad = app.post(
        "/api/predict/classification",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert bad.status_code == 400

    tiny = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    small = app.post(
        "/api/predict/segmentation",
        files={"file": ("tiny.png", _png(tiny), "image/png")},
    )
    assert small.status_code == 400

    empty = app.post(
        "/api/predict/segmentation",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert empty.status_code == 400


def test_inference_without_models_returns_503(tmp_path):
    state = AppState(
        settings=Settings(
            model_dir=tmp_path / "no-models",
            db_path=tmp_path / "monitor.db",
            cors_origins=[],
            max_upload_mb=2,
            device="cpu",
        ),
        predictor=Predictor(tmp_path / "no-models", device="cpu"),
        store=MonitoringStore(tmp_path / "monitor.db"),
        references={},
    )
    application = create_app()
    application.state.services = state
    with TestClient(application) as client:
        image = _png(Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)))
        response = client.post(
            "/api/predict/segmentation", files={"file": ("case.png", image, "image/png")}
        )
        assert response.status_code == 503
        assert "train one first" in response.json()["detail"]


def test_demo_endpoint_optional(app):
    response = app.get("/api/demo/segmentation")
    assert response.status_code in {200, 404}
    if response.status_code == 200:
        body = response.json()
        assert body["task"] == "segmentation"
        assert base64.b64decode(body["image_base64"])[:4] == b"\x89PNG"

    bad = app.get("/api/demo/unknown")
    assert bad.status_code == 400
