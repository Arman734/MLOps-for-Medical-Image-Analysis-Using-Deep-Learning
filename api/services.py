"""Application state: settings, predictor, monitoring store, reference stats."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from medmlops.config import PROJECT_ROOT, Env
from medmlops.data.common import load_reference_stats
from medmlops.monitor import MonitoringStore
from medmlops.predictor import Predictor


@dataclass
class Settings:
    model_dir: Path
    db_path: Path
    cors_origins: list[str]
    max_upload_mb: int
    device: str


def load_settings() -> Settings:
    env = Env()
    return Settings(
        model_dir=PROJECT_ROOT / (env.get("MEDMLOPS_MODEL_DIR", "artifacts/models") or "artifacts/models"),
        db_path=PROJECT_ROOT / (env.get("MEDMLOPS_DB_PATH", "artifacts/monitoring.db") or "artifacts/monitoring.db"),
        cors_origins=[o.strip() for o in (env.get("MEDMLOPS_CORS_ORIGINS", "") or "").split(",") if o.strip()]
        or ["http://localhost:5173", "http://127.0.0.1:5173"],
        max_upload_mb=int(env.get("MEDMLOPS_MAX_UPLOAD_MB", "15") or 15),
        device=env.get("MEDMLOPS_DEVICE", "auto") or "auto",
    )


def find_reference_stats() -> dict[str, dict]:
    """Locate drift baselines produced by the data-prep stage."""
    references: dict[str, dict] = {}
    isic = PROJECT_ROOT / "data" / "processed" / "isic2018" / "reference_stats.json"
    if isic.exists():
        references["segmentation"] = load_reference_stats(isic)
    medmnist_root = PROJECT_ROOT / "data" / "processed" / "medmnist"
    if medmnist_root.exists():
        candidates = sorted(medmnist_root.glob("*/reference_stats.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            references["classification"] = load_reference_stats(candidates[0])
    return references


@dataclass
class AppState:
    settings: Settings
    predictor: Predictor
    store: MonitoringStore
    references: dict[str, dict] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    @property
    def uptime_s(self) -> float:
        return time.time() - self.started_at

    def dataset_summaries(self) -> dict:
        from medmlops.config import load_config
        from medmlops.data import isic, medmnist

        out = {}
        for name, path in (("segmentation", "configs/isic_unet.yaml"), ("classification", "configs/pneumonia_densenet.yaml")):
            try:
                cfg = load_config(path)
                summary = isic.dataset_summary(cfg) if cfg.data.name == "isic2018" else medmnist.dataset_summary(cfg)
                out[name] = summary
            except Exception as exc:  # pragma: no cover - defensive
                out[name] = {"status": f"error: {exc}"}
        return out


def build_state() -> AppState:
    settings = load_settings()
    predictor = Predictor(settings.model_dir, device=settings.device)
    predictor.refresh()
    store = MonitoringStore(settings.db_path)
    references = find_reference_stats()
    return AppState(settings=settings, predictor=predictor, store=store, references=references)


def json_safe(data):
    """Replace NaN/inf with None so JSON responses stay valid."""
    import math

    if isinstance(data, dict):
        return {k: json_safe(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [json_safe(v) for v in data]
    if isinstance(data, float):
        return data if math.isfinite(data) else None
    return data
