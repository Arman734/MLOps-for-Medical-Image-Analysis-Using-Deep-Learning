"""Tracking backend selection, DagsHub guardrails and registry mirroring."""

from __future__ import annotations

import os
import types

import pytest

from medmlops import tracking


class _EmptyEnv:
    def get(self, key, default=None):
        return default


class _OsEnv:
    def get(self, key, default=None):
        return os.environ.get(key, default)


def test_artifact_location_follows_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "store"))
    local = tracking.experiment_artifact_location("sqlite:///mlflow.db")
    assert local == str((tmp_path / "store").resolve())
    assert tracking.experiment_artifact_location("file:///tmp/mlruns") == local
    assert tracking.experiment_artifact_location("http://mlflow:5000") is None
    assert tracking.experiment_artifact_location("sqlite:///mlflow.db", dagshub_enabled=True) is None
    assert tracking.experiment_artifact_location("https://dagshub.com/u/r.mlflow", dagshub_enabled=True) is None


def test_registry_names_are_env_overridable(monkeypatch):
    monkeypatch.delenv("MEDMLOPS_REGISTRY_NAMES", raising=False)
    assert tracking.registry_names() == ["isic-lesion-unet", "pneumonia-densenet"]
    monkeypatch.setenv("MEDMLOPS_REGISTRY_NAMES", " unet , ,clf ")
    assert tracking.registry_names() == ["unet", "clf"]


def test_dagshub_requires_credentials(monkeypatch):
    monkeypatch.setattr(tracking, "Env", _EmptyEnv)
    cfg = types.SimpleNamespace(experiment="unit")
    with pytest.raises(RuntimeError, match="DAGSHUB_USER_NAME"):
        tracking.init_tracking(cfg, dagshub=True)


def test_dagshub_uri_gets_basic_auth(monkeypatch):
    monkeypatch.setattr(tracking, "Env", _OsEnv)
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    monkeypatch.setenv("DAGSHUB_USER_NAME", "alice")
    monkeypatch.setenv("DAGSHUB_USER_TOKEN", "token-123")

    tracking._apply_remote_auth("https://dagshub.com/alice/medmlops.mlflow")
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "alice"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "token-123"

    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME")
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD")
    tracking._apply_remote_auth("http://mlflow:5000")
    assert "MLFLOW_TRACKING_USERNAME" not in os.environ


def test_download_champion_copies_checkpoint(tmp_path, monkeypatch):
    import mlflow

    artifact_dir = tmp_path / "run" / "checkpoints"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "seg.pt").write_bytes(b"weights")
    (artifact_dir / "seg.json").write_text('{"task": "segmentation"}', encoding="utf-8")

    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            assert (name, alias) == ("isic-lesion-unet", "champion")
            return types.SimpleNamespace(version="7", run_id="run-7")

    monkeypatch.setattr(tracking, "ensure_tracking", lambda: "sqlite:///mlflow.db")
    monkeypatch.setattr(mlflow, "MlflowClient", lambda *args, **kwargs: FakeClient())
    monkeypatch.setattr(
        mlflow.artifacts, "download_artifacts", lambda **kwargs: str(artifact_dir), raising=False
    )

    result = tracking.download_champion("isic-lesion-unet", tmp_path / "models")
    target = tmp_path / "models" / "isic-lesion-unet.pt"
    assert target.read_bytes() == b"weights"
    assert target.with_suffix(".json").read_text(encoding="utf-8") == '{"task": "segmentation"}'
    assert result["version"] == "7"
    assert result["run_id"] == "run-7"


def test_sync_champions_is_best_effort(tmp_path, monkeypatch):
    def fake_download(name, dest):
        if name == "bad":
            raise LookupError("no champion")
        return {"registry_name": name, "checkpoint": str(dest)}

    monkeypatch.setattr(tracking, "download_champion", fake_download)
    result = tracking.sync_champions(tmp_path, names=["good", "bad"])
    assert [item["registry_name"] for item in result["downloaded"]] == ["good"]
    assert result["errors"] == [{"registry_name": "bad", "error": "no champion"}]
