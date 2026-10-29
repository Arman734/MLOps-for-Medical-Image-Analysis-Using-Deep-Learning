"""Drift detection and the SQLite prediction ledger."""

from __future__ import annotations

import csv
import json
import sqlite3
import time

import numpy as np

from medmlops.data.common import FEATURE_NAMES
from medmlops.monitor import MonitoringStore, drift_status, population_stability_index


def _reference(rng, n: int = 200, shift: float = 0.0) -> tuple[dict, np.ndarray]:
    values = rng.normal(50 + shift, 5, n)
    counts, edges = np.histogram(values, bins=12)
    reference = {
        "features": {
            "brightness": {
                "mean": float(values.mean()),
                "bin_edges": edges.tolist(),
                "expected_freq": (counts / counts.sum()).tolist(),
            }
        }
    }
    return reference, values


def test_psi_zero_for_identical_distribution(rng):
    reference, values = _reference(rng)
    psi = population_stability_index(
        reference["features"]["brightness"]["expected_freq"],
        reference["features"]["brightness"]["bin_edges"],
        values.tolist(),
    )
    assert psi < 1e-3
    assert drift_status(psi) == "stable"


def test_psi_detects_shift(rng):
    reference, _ = _reference(rng)
    _, shifted = _reference(rng, shift=20)
    psi = population_stability_index(
        reference["features"]["brightness"]["expected_freq"],
        reference["features"]["brightness"]["bin_edges"],
        shifted.tolist(),
    )
    assert psi > 0.25
    assert drift_status(psi) == "alert"


def test_psi_handles_empty_inputs():
    assert population_stability_index([], [], []) == 0.0
    assert drift_status(0.05) == "stable"
    assert drift_status(0.15) == "watch"
    assert drift_status(0.4) == "alert"


def _features(value: float) -> dict:
    return {name: value for name in FEATURE_NAMES}


def test_store_ledger_feedback_and_export(tmp_path, feature_reference):
    store = MonitoringStore(tmp_path / "monitor.db")
    pid = store.log_prediction(
        task="segmentation",
        model_id="unet",
        model_version="run-1",
        filename="case.png",
        latency_ms=123.4,
        label="lesion",
        confidence=0.87,
        features=_features(4.2),
        meta={"threshold": 0.5},
    )
    store.log_prediction(
        task="classification",
        model_id="densenet",
        model_version="run-2",
        filename="case2.png",
        latency_ms=50.0,
        label="pneumonia",
        confidence=0.91,
        features=_features(3.1),
    )
    store.add_feedback(pid, "correct", "matches radiology read")

    assert store.count() == 2
    assert store.count("segmentation") == 1
    assert store.label_distribution("classification") == {"pneumonia": 1}

    recent = store.recent(limit=5)
    assert recent[0]["task"] == "classification"
    assert recent[1]["feedback"]["verdict"] == "correct"

    summary = store.summary({"segmentation": feature_reference})
    assert summary["total_predictions"] == 2
    assert summary["by_task"]["segmentation"]["predictions"] == 1
    assert summary["feedback"]["correct"] == 1
    assert summary["drift"]["segmentation"]["status"] == "insufficient_data"
    assert summary["drift"]["segmentation"]["window"] == 1

    path = store.export_csv(tmp_path / "export.csv")
    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert rows[0][0] == "id"
    assert rows[0][-2:] == ["verdict", "comment"]
    assert len(rows) == 3
    by_task = {row[2]: row for row in rows[1:]}
    assert by_task["segmentation"][-2] == "correct"
    assert by_task["segmentation"][-1] == "matches radiology read"
    assert by_task["classification"][-2:] == ["", ""]


def test_store_uses_wal_indexes_and_maintenance(tmp_path):
    db_path = tmp_path / "maintenance.db"
    store = MonitoringStore(db_path)
    stats = store.stats()
    assert stats["wal"] is True
    assert stats["journal_mode"] == "wal"

    with sqlite3.connect(db_path) as conn:
        indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"idx_predictions_ts", "idx_predictions_task_ts", "idx_feedback_prediction"} <= indexes

    pid = store.log_prediction(
        task="classification",
        model_id="densenet",
        model_version="run-1",
        filename="old.png",
        latency_ms=12.0,
        label="pneumonia",
        confidence=0.7,
        features={"brightness": 3.0},
        meta={"margin": 0.4},
    )
    store.add_feedback(pid, "incorrect", "needs review")
    assert store.prediction_exists(pid)
    detail = store.get_prediction(pid)
    assert detail is not None
    assert detail["features"]["brightness"] == 3.0
    assert detail["feedback"][0]["verdict"] == "incorrect"
    assert store.get_prediction("missing") is None

    backup_path = store.backup(tmp_path / "backup.db")
    backup = MonitoringStore(backup_path)
    assert backup.count() == 1
    backup.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE predictions SET ts = ? WHERE id = ?", (time.time() - 40 * 86400, pid))
    result = store.prune(30)
    assert result["removed_predictions"] == 1
    assert result["removed_feedback"] == 1
    assert store.count() == 0
    store.close()


def test_drift_requires_min_window_then_tracks_shift(tmp_path, feature_reference, rng):
    from medmlops.monitor import _MIN_DRIFT_SAMPLES

    ref = feature_reference["features"]

    def draw(name: str) -> float:
        freq = np.asarray(ref[name]["expected_freq"])
        edges = np.asarray(ref[name]["bin_edges"])
        index = int(rng.choice(len(freq), p=freq / freq.sum()))
        return float(rng.uniform(edges[index], edges[index + 1]))

    def log(store, sample, count):
        for _ in range(count):
            store.log_prediction(
                task="classification",
                model_id="densenet",
                model_version="run-1",
                filename=None,
                latency_ms=20.0,
                label="pneumonia",
                confidence=0.8,
                features={name: sample(name) for name in FEATURE_NAMES},
            )

    stable_store = MonitoringStore(tmp_path / "stable.db")
    log(stable_store, draw, _MIN_DRIFT_SAMPLES * 2)
    stable = stable_store.drift(feature_reference, "classification")
    assert stable["status"] in {"stable", "watch"}
    assert set(stable["features"]) == set(FEATURE_NAMES)

    shifted_store = MonitoringStore(tmp_path / "shifted.db")
    log(
        shifted_store,
        lambda name: float(
            rng.normal(ref[name]["mean"] + 6 * max(ref[name]["std"], 1e-3), max(ref[name]["std"], 1e-3))
        ),
        _MIN_DRIFT_SAMPLES,
    )
    drifted = shifted_store.drift(feature_reference, "classification")
    assert drifted["status"] == "alert"
    assert drifted["window"] == _MIN_DRIFT_SAMPLES


def test_store_without_reference(tmp_path):
    store = MonitoringStore(tmp_path / "empty.db")
    assert store.drift(None, "segmentation")["status"] == "no_reference"
    summary = store.summary()
    assert summary["total_predictions"] == 0
    assert summary["drift"]["segmentation"]["status"] == "no_reference"


def test_reference_stats_persisted(tmp_path, feature_reference):
    from medmlops.data.common import load_reference_stats, save_reference_stats

    path = save_reference_stats(feature_reference, tmp_path / "ref.json")
    loaded = load_reference_stats(path)
    assert loaded["n_samples"] == 40
    assert set(loaded["features"]) == set(FEATURE_NAMES)
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["dataset"] == "test"
