"""End-to-end smoke test against a running MedMLOps API.

Usage:
    python scripts/smoke_test.py [--base-url http://127.0.0.1:8000] [--api-key KEY]

The API key defaults to ``MEDMLOPS_API_KEY`` from the environment (or .env), so
the same script works against local and protected deployments.

Exits non-zero when a required endpoint is unhealthy. Model-dependent checks
are skipped (with a warning) when no checkpoint has been trained yet.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time

import httpx

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{status}] {name}{f' - {detail}' if detail else ''}")


def check(client: httpx.Client, path: str, *, required: bool = True, timeout: float = 60.0) -> httpx.Response | None:
    try:
        response = client.get(path, timeout=timeout)
    except httpx.HTTPError as exc:
        record(FAIL if required else WARN, path, f"request failed: {exc}")
        return None
    if response.status_code == 200:
        record(PASS, path, f"{len(response.content)} bytes")
        return response
    record(FAIL if required else WARN, path, f"HTTP {response.status_code}")
    return None


def demo_image(client: httpx.Client, task: str) -> bytes | None:
    """A held-out demo case when datasets are present, else a synthetic image."""
    response = client.get(f"/api/demo/{task}", timeout=30.0)
    if response.status_code == 200:
        record(PASS, f"/api/demo/{task}", "held-out case")
        return base64.b64decode(response.json()["image_base64"])
    record(WARN, f"/api/demo/{task}", "datasets not prepared - using a synthetic image")
    import io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(7)
    array = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def predict(client: httpx.Client, task: str) -> dict | None:
    image = demo_image(client, task)
    if image is None:
        return None
    fields = {"tta": "true"} if task == "segmentation" else {"explain": "true"}
    started = time.perf_counter()
    try:
        response = client.post(
            f"/api/predict/{task}",
            files={"file": (f"demo_{task}.png", image, "image/png")},
            data=fields,
            timeout=180.0,
        )
    except httpx.HTTPError as exc:
        record(FAIL, f"POST /api/predict/{task}", f"request failed: {exc}")
        return None
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        record(FAIL, f"POST /api/predict/{task}", f"HTTP {response.status_code}: {response.text[:200]}")
        return None
    body = response.json()
    if task == "segmentation":
        if not body.get("overlay_base64"):
            record(FAIL, "segmentation overlay", "missing overlay_base64")
            return None
        detail = f"lesion_fraction={body['lesion_fraction']:.4f} latency={body['latency_ms']:.1f}ms wall={elapsed:.2f}s"
    else:
        detail = f"{body['predicted_label']} p={body['confidence']:.3f} latency={body['latency_ms']:.1f}ms wall={elapsed:.2f}s"
    record(PASS, f"POST /api/predict/{task}", detail)

    feedback = client.post(
        "/api/feedback",
        json={"prediction_id": body["prediction_id"], "verdict": "correct", "comment": "smoke test"},
        timeout=30.0,
    )
    record(PASS if feedback.status_code == 200 else FAIL, "POST /api/feedback", f"HTTP {feedback.status_code}")

    detail_response = client.get(f"/api/predictions/{body['prediction_id']}", timeout=30.0)
    if detail_response.status_code == 200:
        record(PASS, "GET /api/predictions/{id}", "ledger row + feedback")
    else:
        record(FAIL, "GET /api/predictions/{id}", f"HTTP {detail_response.status_code}")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.environ.get("MEDMLOPS_API_KEY", ""))
    args = parser.parse_args()

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    with httpx.Client(base_url=args.base_url, follow_redirects=True, headers=headers) as client:
        health = check(client, "/api/health")
        ready: dict = {}
        if health is not None:
            ready = health.json()
            record(
                PASS if ready.get("database_ok") else FAIL,
                "health.database_ok",
                "SQLite ledger reachable" if ready.get("database_ok") else "ledger ping failed",
            )
        probe = client.get("/api/ready", timeout=30.0)
        if probe.status_code == 200:
            record(PASS, "/api/ready", probe.json().get("status", ""))
        elif probe.status_code == 503:
            record(WARN, "/api/ready", "models or ledger not ready yet")
        else:
            record(FAIL, "/api/ready", f"HTTP {probe.status_code}")
        check(client, "/api/system")
        models = check(client, "/api/models")
        check(client, "/api/pipeline/configs")
        check(client, "/api/experiments", required=False)
        check(client, "/api/monitoring/summary")
        storage = check(client, "/api/monitoring/storage")
        if storage is not None:
            info = storage.json()
            record(
                PASS if info.get("wal") else WARN,
                "ledger WAL",
                f"{info.get('predictions', 0)} predictions · {info.get('size_bytes', 0)} bytes",
            )
        export = check(client, "/api/monitoring/export.csv")
        if export is not None:
            header = export.text.splitlines()[0] if export.text else ""
            record(
                PASS if "verdict" in header else FAIL,
                "export.csv schema",
                header[:80],
            )
        metrics = check(client, "/metrics")
        if metrics is not None and "medmlops_api_requests_total" not in metrics.text:
            record(FAIL, "/metrics content", "missing medmlops_api_requests_total")
        check(client, "/", required=False)

        if ready.get("segmentation_ready"):
            predict(client, "segmentation")
        else:
            record(WARN, "segmentation inference", "no checkpoint - run `make train-seg`")

        if ready.get("classification_ready"):
            predict(client, "classification")
        else:
            record(WARN, "classification inference", "no checkpoint - run `make train-clf`")

        if models is not None:
            payload = models.json()
            loaded = len(payload.get("models", []))
            record(PASS if loaded else WARN, "models loaded", str(loaded))

    failures = [row for row in results if row[0] == FAIL]
    warnings = [row for row in results if row[0] == WARN]
    print(f"\n{len(results) - len(failures) - len(warnings)} passed, {len(warnings)} warnings, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
