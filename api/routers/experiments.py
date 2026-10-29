"""Experiment tracking, registry and comparison endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from medmlops import tracking

router = APIRouter(prefix="/api", tags=["experiments"])


@router.get("/experiments")
def list_experiments(limit: int = Query(30, ge=1, le=200)) -> dict:
    try:
        runs = tracking.search_runs(limit=limit)
    except Exception as exc:  # pragma: no cover - mlflow backend issues
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}") from exc
    return {"runs": runs, "tracking": tracking.tracking_info()}


@router.get("/experiments/{run_id}")
def experiment_detail(run_id: str) -> dict:
    run = tracking.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return run


@router.get("/registry")
def registry() -> dict:
    return {"models": tracking.list_models()}
