"""End-to-end smoke test against a running MedMLOps API.

Usage:
    python scripts/smoke_test.py [--base-url http://127.0.0.1:8000]

Exits non-zero when a required endpoint is unhealthy. Model-dependent checks
are skipped (with a warning) when no checkpoint has been trained yet.
"""

from __future__ import annotations

import argparse
import base64
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


def predict(client: httpx.Client, task: str) -> None:
    demo = check(client, f"/api/demo/{task}")
    if demo is None:
        return
    payload = demo.json()
    image = base64.b64decode(payload["image_base64"])
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
        return
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        record(FAIL, f"POST /api/predict/{task}", f"HTTP {response.status_code}: {response.text[:200]}")
        return
    body = response.json()
    if task == "segmentation":
        if not body.get("overlay_base64"):
            record(FAIL, "segmentation overlay", "missing overlay_base64")
            return
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, follow_redirects=True) as client:
        health = check(client, "/api/health")
        ready: dict = {}
        if health is not None:
            ready = health.json()
        check(client, "/api/system")
        models = check(client, "/api/models")
        check(client, "/api/pipeline/configs")
        check(client, "/api/experiments", required=False)
        check(client, "/api/monitoring/summary")
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
