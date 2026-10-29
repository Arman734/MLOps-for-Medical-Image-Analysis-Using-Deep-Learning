"""MLflow tracking with an opt-in DagsHub backend.

Local default: SQLite backend + local artifact store, no server required.
DagsHub: set MEDMLOPS_TRACKER=dagshub (or pass --dagshub) plus the three
DAGSHUB_* credentials and every run streams to the DagsHub-hosted MLflow.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, Env


def _resolve_uri(uri: str) -> str:
    if uri.startswith("sqlite:///"):
        path = Path(uri.removeprefix("sqlite:///"))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return f"sqlite:///{path.as_posix()}"
    return uri


def _local_artifact_root() -> str:
    raw = os.environ.get("MLFLOW_ARTIFACT_ROOT") or "mlartifacts"
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def experiment_artifact_location(tracking_uri: str, dagshub_enabled: bool = False) -> str | None:
    """Where a new experiment should store artifacts.

    Local SQLite/file stores need an explicit directory. Remote servers and
    DagsHub own their artifact store: passing a client-side path there makes
    uploads fail (the server cannot see the client filesystem).
    """
    if dagshub_enabled:
        return None
    if tracking_uri.startswith(("sqlite:", "file:")):
        return _local_artifact_root()
    return None


def use_dagshub(cfg, dagshub: bool | None = None) -> bool:
    if dagshub is not None:
        return dagshub
    return (Env().get("MEDMLOPS_TRACKER", "local") or "local").strip().lower() == "dagshub"


def ensure_tracking() -> str:
    """Resolve the tracking URI for read paths (API, CLI) without a training run."""
    import mlflow

    if os.environ.get("_MEDMLOPS_URI_READY") != "1":
        uri = _resolve_uri(os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")
        os.environ["MLFLOW_TRACKING_URI"] = uri
        mlflow.set_tracking_uri(uri)
        os.environ["_MEDMLOPS_URI_READY"] = "1"
    return mlflow.get_tracking_uri()


def promote_by_run(registry_name: str, run_id: str) -> dict:
    import mlflow

    ensure_tracking()
    client = mlflow.MlflowClient()
    version = _find_version(client, registry_name, run_id)
    if version is None:
        raise LookupError(f"no version of '{registry_name}' was produced by run {run_id}")
    client.set_registered_model_alias(name=registry_name, alias="champion", version=version)
    client.set_model_version_tag(registry_name, version, "promoted_at", str(int(__import__("time").time())))
    return {"name": registry_name, "version": version, "run_id": run_id, "alias": "champion"}


def init_tracking(cfg, dagshub: bool | None = None) -> str:
    """Point MLflow at the configured backend and activate the experiment."""
    import mlflow

    use_dagshub_backend = use_dagshub(cfg, dagshub)
    if use_dagshub_backend:
        env = Env()
        owner = env.get("DAGSHUB_USER_NAME")
        repo = env.get("DAGSHUB_REPO_NAME")
        token = env.get("DAGSHUB_USER_TOKEN")
        if not (owner and repo and token):
            raise RuntimeError(
                "DagsHub tracking requested but DAGSHUB_USER_NAME / DAGSHUB_REPO_NAME / "
                "DAGSHUB_USER_TOKEN are not set (see .env.example)"
            )
        os.environ["DAGSHUB_USER_TOKEN"] = token
        import dagshub

        dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
        resolved = mlflow.get_tracking_uri()
    else:
        resolved = _resolve_uri(os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(resolved)

    os.environ["MLFLOW_TRACKING_URI"] = resolved
    os.environ["_MEDMLOPS_URI_READY"] = "1"
    artifact_location = experiment_artifact_location(resolved, use_dagshub_backend)
    if artifact_location:
        os.environ.setdefault("MLFLOW_ARTIFACT_ROOT", artifact_location)

    experiment_name = str(cfg.experiment)
    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        if artifact_location:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        else:
            mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    return mlflow.get_tracking_uri()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - git may be missing
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10
        )
        return bool(out.stdout.strip())
    except Exception:  # pragma: no cover
        return False


def log_json(data: Any, artifact_file: str) -> None:
    import mlflow

    mlflow.log_dict(data, artifact_file)


def log_figure(fig, artifact_file: str) -> None:
    import mlflow

    mlflow.log_figure(fig, artifact_file)


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    import mlflow

    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def register_model(
    model,
    cfg,
    input_example,
    description: str,
    metrics: dict[str, float],
    run_id: str,
) -> dict:
    """Log the torch model to the active run and register a new version."""
    import copy

    import mlflow
    import mlflow.pytorch
    from mlflow.models import infer_signature

    device = next(model.parameters()).device
    with _torch_inference_mode(model):
        example_out = model(input_example.to(device))
    signature = infer_signature(input_example.numpy(), example_out.detach().cpu().numpy())

    registry_name = str(cfg.mlflow.get("registry_model_name", cfg.experiment))
    cpu_model = copy.deepcopy(model).to("cpu")
    info = mlflow.pytorch.log_model(
        pytorch_model=cpu_model,
        name="model",
        signature=signature,
        input_example=input_example.numpy(),
        registered_model_name=registry_name,
    )
    client = mlflow.MlflowClient()
    version = _find_version(client, registry_name, run_id)
    if version is not None:
        client.update_model_version(
            name=registry_name,
            version=version,
            description=description,
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value == value:
                client.set_model_version_tag(registry_name, version, f"metric.{key}", str(round(float(value), 4)))
        client.set_model_version_tag(registry_name, version, "run_id", run_id)
    return {
        "registry_name": registry_name,
        "version": version,
        "model_uri": info.model_uri,
        "signature": signature.to_dict(),
    }


class _torch_inference_mode:
    def __init__(self, model) -> None:
        self.model = model

    def __enter__(self):
        import torch

        module = self.model
        self._was_training = module.training
        module.eval()
        self._ctx = torch.no_grad()
        return self._ctx.__enter__()

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        self.model.train(self._was_training)
        return False


def _find_version(client, registry_name: str, run_id: str) -> str | None:
    versions = client.search_model_versions(f"name='{registry_name}'")
    matches = [v for v in versions if v.run_id == run_id]
    if not matches:
        return None
    return max(matches, key=lambda v: int(v.version)).version


def promote_to_champion(cfg, version: str) -> None:
    import mlflow

    name = str(cfg.mlflow.get("registry_model_name", cfg.experiment))
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name=name, alias="champion", version=version)


def promote_run(cfg, run_id: str) -> tuple[str, str]:
    """Point the ``champion`` alias at the version produced by ``run_id``."""
    import mlflow

    name = str(cfg.mlflow.get("registry_model_name", cfg.experiment))
    client = mlflow.MlflowClient()
    version = _find_version(client, name, run_id)
    if version is None:
        raise LookupError(f"no registered version found for run {run_id} in model '{name}'")
    promote_to_champion(cfg, version)
    return name, version


def get_champion(cfg) -> dict | None:
    import mlflow

    ensure_tracking()

    name = str(cfg.mlflow.get("registry_model_name", cfg.experiment))
    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(name, "champion")
    except Exception:
        return None
    run = client.get_run(version.run_id)
    return {
        "name": name,
        "version": version.version,
        "run_id": version.run_id,
        "metrics": run.data.metrics,
        "description": version.description,
        "status": version.status,
        "tags": dict(version.tags or {}),
    }


def maybe_promote(cfg, run_id: str, metric_key: str, metric_value: float) -> bool:
    """Promote the freshly registered version when it beats the champion."""
    import mlflow

    try:
        name = str(cfg.mlflow.get("registry_model_name", cfg.experiment))
        client = mlflow.MlflowClient()
        version = _find_version(client, name, run_id)
        if version is None:
            return False
        champion = get_champion(cfg)
        if champion is None:
            promote_to_champion(cfg, version)
            mlflow.set_tag("promoted_to_champion", "true (first version)")
            return True
        if metric_value > float(champion["metrics"].get(metric_key, float("-inf"))):
            promote_to_champion(cfg, version)
            mlflow.set_tag("promoted_to_champion", f"true (beat v{champion['version']})")
            return True
        mlflow.set_tag("promoted_to_champion", f"false (champion v{champion['version']} better)")
        return False
    except Exception as exc:  # pragma: no cover - registry optional
        mlflow.set_tag("promoted_to_champion", f"error: {exc}")
        return False


def list_models() -> list[dict]:
    import mlflow

    ensure_tracking()

    client = mlflow.MlflowClient()
    out: list[dict] = []
    try:
        registered = client.search_registered_models()
    except Exception:  # pragma: no cover - empty registry store
        return out
    for model in registered:
        versions = []
        for v in sorted(model.latest_versions, key=lambda mv: int(mv.version), reverse=True):
            tags = dict(v.tags or {})
            versions.append(
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id,
                    "created": v.creation_timestamp,
                    "metrics": {k.removeprefix("metric."): float(val) for k, val in tags.items() if k.startswith("metric.")},
                }
            )
        try:
            champion = client.get_model_version_by_alias(model.name, "champion").version
        except Exception:
            champion = None
        out.append({"name": model.name, "champion": champion, "versions": versions})
    return out


def search_runs(experiment_names: list[str] | None = None, limit: int = 50) -> list[dict]:
    import mlflow

    ensure_tracking()

    names = experiment_names
    if not names:
        names = [experiment.name for experiment in mlflow.search_experiments()]
    if not names:
        return []
    try:
        runs = mlflow.search_runs(
            experiment_names=names,
            order_by=["attributes.start_time DESC"],
            max_results=limit,
            output_format="list",
        )
    except Exception:
        return []
    records: list[dict] = []
    for run in runs or []:
        info, data = run.info, run.data
        start_ms, end_ms = info.start_time, info.end_time
        records.append(
            {
                "run_id": info.run_id,
                "run_name": data.tags.get("mlflow.runName") or getattr(info, "run_name", None),
                "experiment": info.experiment_id,
                "status": info.status,
                "start_time": round(start_ms / 1000.0, 3) if start_ms else None,
                "duration_s": round((end_ms - start_ms) / 1000.0, 1) if start_ms and end_ms else None,
                "metrics": {name: _clean(value) for name, value in data.metrics.items()},
                "params": dict(data.params),
                "tags": dict(data.tags),
            }
        )
    return records


def _clean(value, ndigits: int = 4):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return round(number, ndigits)


def get_run(run_id: str) -> dict | None:
    import mlflow

    ensure_tracking()

    client = mlflow.MlflowClient()
    try:
        run = client.get_run(run_id)
        history = {}
        for key in ("train_loss", "val_loss", "val_dice", "val_auc", "val_acc", "lr"):
            points = client.get_metric_history(run_id, key)
            if points:
                history[key] = [{"step": p.step, "value": p.value, "timestamp": p.timestamp} for p in points]
        return {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "status": run.info.status,
            "experiment_id": run.info.experiment_id,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
            "history": history,
        }
    except Exception:
        return None


def experiment_names() -> list[str]:
    import mlflow

    ensure_tracking()

    try:
        return [e.name for e in mlflow.search_experiments()]
    except Exception:
        return []


def tracking_info() -> dict:
    import mlflow

    ensure_tracking()

    uri = mlflow.get_tracking_uri()
    local = uri.startswith(("sqlite:", "file:"))
    return {
        "tracking_uri": uri,
        "backend": "dagshub" if "dagshub" in uri else ("local" if local else "server"),
        "ui_url": None if local else uri,
        "artifact_root": _local_artifact_root() if local else None,
        "experiments": experiment_names(),
    }


def log_artifact_tree(directory: str | Path, artifact_path: str) -> None:
    import mlflow

    mlflow.log_artifacts(str(directory), artifact_path=artifact_path)


def save_run_summary(run_id: str, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"run_id": run_id}, fh, indent=2)


def registry_names() -> list[str]:
    """Registry names that a deployment should mirror (env overridable)."""
    raw = Env().get("MEDMLOPS_REGISTRY_NAMES") or "isic-lesion-unet,pneumonia-densenet"
    return [name.strip() for name in raw.split(",") if name.strip()]


def download_champion(registry_name: str, dest_dir: str | Path) -> dict:
    """Copy the champion run's checkpoint + metadata out of MLflow into a model dir.

    Training logs ``checkpoints/*.pt`` and ``*.json`` as run artifacts, so a
    fresh deployment can bootstrap serving state straight from the registry
    (local SQLite, an MLflow server or DagsHub) without retraining.
    """
    import mlflow

    ensure_tracking()
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(registry_name, "champion")
    artifact_dir = Path(mlflow.artifacts.download_artifacts(run_id=version.run_id, artifact_path="checkpoints"))
    checkpoints = sorted(artifact_dir.glob("*.pt"))
    if not checkpoints:
        raise LookupError(f"run {version.run_id} has no checkpoint artifact for '{registry_name}'")
    source = checkpoints[0]
    sidecar = source.with_suffix(".json")
    if not sidecar.exists():
        raise LookupError(f"checkpoint {source.name} has no metadata sidecar")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in registry_name)
    target = dest_dir / f"{safe}.pt"
    shutil.copy2(source, target)
    shutil.copy2(sidecar, target.with_suffix(".json"))
    return {
        "registry_name": registry_name,
        "version": version.version,
        "run_id": version.run_id,
        "checkpoint": str(target),
        "metadata": str(target.with_suffix(".json")),
    }


def sync_champions(dest_dir: str | Path, names: list[str] | None = None) -> dict:
    """Mirror every available champion into the serving directory (best effort)."""
    downloaded: list[dict] = []
    errors: list[dict] = []
    for name in names or registry_names():
        try:
            downloaded.append(download_champion(name, dest_dir))
        except Exception as exc:  # registry may be empty or unreachable
            errors.append({"registry_name": name, "error": str(exc)})
    return {"downloaded": downloaded, "errors": errors}
