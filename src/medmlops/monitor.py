"""Prediction logging, clinician feedback and data-drift detection."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .data.common import FEATURE_NAMES

_PSI_WARN = 0.1
_PSI_ALERT = 0.25
_MIN_DRIFT_SAMPLES = 150


def population_stability_index(
    expected_freq: list[float], bin_edges: list[float], values: list[float], alpha: float = 0.5
) -> float:
    if not expected_freq or not bin_edges or not values:
        return 0.0
    edges = np.asarray(bin_edges, dtype=np.float64)
    expected = np.asarray(expected_freq, dtype=np.float64)
    expected = expected / max(expected.sum(), 1e-9)
    clipped = np.clip(np.asarray(values, dtype=np.float64), edges[0] + 1e-9, edges[-1] - 1e-9)
    counts, _ = np.histogram(clipped, bins=edges)
    actual = counts / max(counts.sum(), 1)
    # Laplace smoothing keeps sparse bins (and small windows) from producing
    # absurd PSI values when a single sample lands in a 0.4% reference bin.
    n_bins = len(expected)
    prior = alpha / n_bins
    expected = (expected + prior) / (1.0 + alpha)
    actual = (actual + prior) / (1.0 + alpha)
    psi = np.sum((actual - expected) * np.log(actual / expected))
    return float(max(psi, 0.0))


def drift_status(psi: float) -> str:
    if psi >= _PSI_ALERT:
        return "alert"
    if psi >= _PSI_WARN:
        return "watch"
    return "stable"


class MonitoringStore:
    """SQLite-backed prediction ledger (safe for FastAPI's threadpool)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                ts REAL NOT NULL,
                task TEXT NOT NULL,
                model_id TEXT,
                model_version TEXT,
                filename TEXT,
                latency_ms REAL,
                label TEXT,
                confidence REAL,
                features TEXT,
                meta TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT NOT NULL,
                verdict TEXT NOT NULL,
                comment TEXT,
                ts REAL NOT NULL
            );
            """
        )
        self._conn.commit()

    def log_prediction(
        self,
        *,
        task: str,
        model_id: str | None,
        model_version: str | None,
        filename: str | None,
        latency_ms: float,
        label: str | None,
        confidence: float | None,
        features: dict | None = None,
        meta: dict | None = None,
        prediction_id: str | None = None,
    ) -> str:
        pid = prediction_id or uuid.uuid4().hex[:12]
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO predictions (id, ts, task, model_id, model_version, filename, latency_ms, label, confidence, features, meta)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pid,
                    time.time(),
                    task,
                    model_id,
                    model_version,
                    filename,
                    float(latency_ms),
                    label,
                    float(confidence) if confidence is not None else None,
                    json.dumps(features or {}),
                    json.dumps(meta or {}),
                ),
            )
        return pid

    def add_feedback(self, prediction_id: str, verdict: str, comment: str = "") -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO feedback (prediction_id, verdict, comment, ts) VALUES (?, ?, ?, ?)",
                (prediction_id, verdict, comment, time.time()),
            )

    def _rows(self, limit: int, task: str | None = None) -> list[sqlite3.Row]:
        query = "SELECT * FROM predictions"
        params: list = []
        if task:
            query += " WHERE task = ?"
            params.append(task)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return list(self._conn.execute(query, params))

    def recent(self, limit: int = 25, task: str | None = None) -> list[dict]:
        rows = self._rows(limit, task)
        with self._lock:
            feedback = {
                row["prediction_id"]: {"verdict": row["verdict"], "comment": row["comment"]}
                for row in self._conn.execute("SELECT * FROM feedback ORDER BY ts DESC LIMIT 500")
            }
        out = []
        for row in rows:
            out.append(
                {
                    "id": row["id"],
                    "ts": row["ts"],
                    "task": row["task"],
                    "model_id": row["model_id"],
                    "model_version": row["model_version"],
                    "filename": row["filename"],
                    "latency_ms": row["latency_ms"],
                    "label": row["label"],
                    "confidence": row["confidence"],
                    "feedback": feedback.get(row["id"]),
                }
            )
        return out

    def count(self, task: str | None = None) -> int:
        query = "SELECT COUNT(*) AS n FROM predictions"
        params: tuple = ()
        if task:
            query += " WHERE task = ?"
            params = (task,)
        with self._lock:
            return int(self._conn.execute(query, params).fetchone()["n"])

    def label_distribution(self, task: str) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT label, COUNT(*) AS n FROM predictions WHERE task = ? GROUP BY label", (task,)
            ).fetchall()
        return {str(row["label"]): int(row["n"]) for row in rows}

    def drift(self, reference: dict | None, task: str, window: int = 200) -> dict:
        if not reference:
            return {"status": "no_reference", "features": {}, "window": 0}
        rows = self._rows(window, task)
        if len(rows) < _MIN_DRIFT_SAMPLES:
            return {
                "status": "insufficient_data",
                "features": {},
                "window": len(rows),
                "min_samples": _MIN_DRIFT_SAMPLES,
                "has_reference": True,
            }
        features: dict[str, dict] = {}
        statuses: list[str] = []
        for name in FEATURE_NAMES:
            values: list[float] = []
            for row in rows:
                try:
                    payload = json.loads(row["features"] or "{}")
                except json.JSONDecodeError:
                    payload = {}
                if name in payload:
                    values.append(float(payload[name]))
            ref = (reference.get("features") or {}).get(name)
            if not ref or not values:
                continue
            psi = population_stability_index(ref.get("expected_freq", []), ref.get("bin_edges", []), values)
            status = drift_status(psi)
            statuses.append(status)
            features[name] = {
                "psi": round(psi, 4),
                "status": status,
                "reference_mean": round(float(ref.get("mean", 0.0)), 4),
                "current_mean": round(float(np.mean(values)), 4),
                "n": len(values),
            }
        if not statuses:
            overall = "insufficient_data"
        elif "alert" in statuses:
            overall = "alert"
        elif "watch" in statuses:
            overall = "watch"
        else:
            overall = "stable"
        return {"status": overall, "features": features, "window": len(rows)}

    def summary(self, references: dict[str, dict] | None = None, window: int = 200) -> dict:
        references = references or {}
        recent = self.recent(limit=15)
        latencies = [row["latency_ms"] for row in recent if row["latency_ms"]]
        with self._lock:
            agg = self._conn.execute(
                "SELECT task, COUNT(*) AS n, AVG(latency_ms) AS avg_ms,"
                " SUM(CASE WHEN feedback.id IS NOT NULL THEN 1 ELSE 0 END) AS with_feedback"
                " FROM predictions LEFT JOIN feedback ON feedback.prediction_id = predictions.id"
                " GROUP BY task"
            ).fetchall()
            verdicts = self._conn.execute(
                "SELECT verdict, COUNT(*) AS n FROM feedback GROUP BY verdict"
            ).fetchall()
        by_task = {
            row["task"]: {
                "predictions": int(row["n"]),
                "avg_latency_ms": round(float(row["avg_ms"] or 0.0), 1),
                "with_feedback": int(row["with_feedback"] or 0),
            }
            for row in agg
        }
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        return {
            "total_predictions": self.count(),
            "by_task": by_task,
            "recent_latency_p95_ms": round(p95, 1),
            "feedback": {row["verdict"]: int(row["n"]) for row in verdicts},
            "labels": {
                "segmentation": self.label_distribution("segmentation"),
                "classification": self.label_distribution("classification"),
            },
            "drift": {
                task: self.drift(references.get(task), task, window=window)
                for task in ("segmentation", "classification")
            },
            "recent": recent,
        }

    def export_csv(self, path: str | Path) -> Path:
        import csv

        rows = self._rows(limit=100000)
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "timestamp", "task", "model_id", "model_version", "filename", "latency_ms", "label", "confidence"])
            for row in rows:
                writer.writerow(
                    [
                        row["id"],
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["ts"])),
                        row["task"],
                        row["model_id"],
                        row["model_version"],
                        row["filename"],
                        round(row["latency_ms"] or 0, 2),
                        row["label"],
                        round(row["confidence"] or 0, 4),
                    ]
                )
        return path
