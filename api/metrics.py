"""Prometheus instrumentation for the API."""

from __future__ import annotations

import time

from fastapi import Request
from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "medmlops_api_requests_total",
    "HTTP requests served by the MedMLOps API",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "medmlops_api_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
PREDICTION_COUNT = Counter(
    "medmlops_predictions_total",
    "Model predictions served",
    ["task", "label"],
)
PREDICTION_LATENCY = Histogram(
    "medmlops_prediction_duration_seconds",
    "Model inference latency",
    ["task"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
MODEL_READY = Gauge(
    "medmlops_model_ready",
    "Whether a trained model is loaded for the task",
    ["task"],
)


def _route_label(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


async def prometheus_middleware(request: Request, call_next):
    started = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        path = _route_label(request)
        if not path.startswith("/metrics"):
            REQUEST_COUNT.labels(request.method, path, status).inc()
            REQUEST_LATENCY.labels(request.method, path).observe(time.perf_counter() - started)


def refresh_model_gauges(predictor) -> None:
    info = predictor.info()
    MODEL_READY.labels("segmentation").set(1 if info["segmentation_ready"] else 0)
    MODEL_READY.labels("classification").set(1 if info["classification_ready"] else 0)
