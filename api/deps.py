"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from .services import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.services
