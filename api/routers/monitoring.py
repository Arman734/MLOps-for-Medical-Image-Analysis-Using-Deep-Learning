"""Monitoring endpoints: prediction ledger, feedback and distribution drift."""

from __future__ import annotations

import csv
import io
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..deps import get_state
from ..services import AppState, find_reference_stats

router = APIRouter(prefix="/api", tags=["monitoring"])

_TASKS = {"segmentation", "classification"}


def _refresh_missing_references(state: AppState) -> dict[str, dict]:
    """Pick up baselines written by data prep after the server started."""
    if len(state.references) < 2:
        state.references.update(find_reference_stats())
    return state.references


def _task(value: str | None, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if value not in _TASKS:
        raise HTTPException(status_code=400, detail="task must be segmentation or classification")
    return value


@router.get("/monitoring/summary")
def summary(window: int = Query(200, ge=10, le=2000), state: AppState = Depends(get_state)) -> dict:
    return state.store.summary(_refresh_missing_references(state), window=window)


@router.get("/monitoring/recent")
def recent(limit: int = Query(20, ge=1, le=200), task: str | None = None, state: AppState = Depends(get_state)) -> dict:
    return {"predictions": state.store.recent(limit=limit, task=_task(task))}


@router.get("/monitoring/drift")
def drift(
    task: str = Query("segmentation"),
    window: int = Query(200, ge=10, le=2000),
    state: AppState = Depends(get_state),
) -> dict:
    references = _refresh_missing_references(state)
    return state.store.drift(references.get(_task(task, required=True)), task, window=window)


@router.get("/monitoring/storage")
def storage(state: AppState = Depends(get_state)) -> dict:
    """Ledger size, row counts and journal mode for operators."""
    return state.store.stats()


@router.post("/monitoring/prune")
def prune(days: int = Query(30, ge=1, le=3650), state: AppState = Depends(get_state)) -> dict:
    """Delete predictions older than ``days`` (and their orphaned feedback)."""
    return state.store.prune(days)


@router.get("/monitoring/export.csv")
def export_csv(task: str | None = None, state: AppState = Depends(get_state)) -> StreamingResponse:
    rows = state.store.export_rows(task=_task(task))
    filename = f"medmlops_predictions_{task}.csv" if task else "medmlops_predictions.csv"

    def generate():
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "id",
                "timestamp",
                "task",
                "model_id",
                "model_version",
                "filename",
                "latency_ms",
                "label",
                "confidence",
                "verdict",
                "comment",
            ]
        )
        yield buffer.getvalue()
        for row in rows:
            buffer.seek(0)
            buffer.truncate(0)
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
                    row["verdict"],
                    row["comment"],
                ]
            )
            yield buffer.getvalue()

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
