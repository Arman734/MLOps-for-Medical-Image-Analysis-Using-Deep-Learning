"""Pipeline endpoints: launch retraining jobs from the console and stream logs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..deps import get_state
from ..jobs import ALLOWED_CONFIGS, get_manager
from ..metrics import refresh_model_gauges
from ..services import AppState

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


class TrainRequest(BaseModel):
    config: str
    overrides: list[str] = []


def _manager(state: AppState):
    return get_manager(lambda: refresh_model_gauges(state.predictor))


@router.get("/configs")
def available_configs() -> dict:
    return {"configs": sorted(ALLOWED_CONFIGS)}


@router.post("/train")
def start_training(payload: TrainRequest, state: AppState = Depends(get_state)) -> dict:
    if payload.config not in ALLOWED_CONFIGS:
        raise HTTPException(status_code=400, detail=f"config must be one of {sorted(ALLOWED_CONFIGS)}")
    try:
        result = _manager(state).start(payload.config, payload.overrides)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if "error" in result:
        raise HTTPException(status_code=409, detail=result["error"])
    return result


@router.get("/jobs")
def list_jobs(state: AppState = Depends(get_state)) -> dict:
    return {"jobs": _manager(state).list()}


@router.get("/jobs/{job_id}")
def job_detail(job_id: str, offset: int = 0, state: AppState = Depends(get_state)) -> dict:
    job = _manager(state).get(job_id, log_offset=max(0, offset))
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, state: AppState = Depends(get_state)) -> dict:
    job = _manager(state).cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job
