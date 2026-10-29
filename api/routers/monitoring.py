"""Monitoring endpoints: prediction ledger, feedback and distribution drift."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from medmlops.config import PROJECT_ROOT

from ..deps import get_state
from ..services import AppState, find_reference_stats

router = APIRouter(prefix="/api", tags=["monitoring"])


def _refresh_missing_references(state: AppState) -> dict[str, dict]:
    """Pick up baselines written by data prep after the server started."""
    if len(state.references) < 2:
        state.references.update(find_reference_stats())
    return state.references


@router.get("/monitoring/summary")
def summary(window: int = Query(200, ge=10, le=2000), state: AppState = Depends(get_state)) -> dict:
    return state.store.summary(_refresh_missing_references(state), window=window)


@router.get("/monitoring/recent")
def recent(limit: int = Query(20, ge=1, le=200), task: str | None = None, state: AppState = Depends(get_state)) -> dict:
    return {"predictions": state.store.recent(limit=limit, task=task)}


@router.get("/monitoring/drift")
def drift(task: str = Query("segmentation"), window: int = Query(200, ge=10, le=2000), state: AppState = Depends(get_state)) -> dict:
    references = _refresh_missing_references(state)
    return state.store.drift(references.get(task), task, window=window)


@router.get("/monitoring/export.csv")
def export_csv(state: AppState = Depends(get_state)) -> FileResponse:
    path = PROJECT_ROOT / "artifacts" / "monitoring_export.csv"
    state.store.export_csv(path)
    return FileResponse(path, media_type="text/csv", filename="medmlops_predictions.csv")
