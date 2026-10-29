"""FastAPI application: inference API, MLOps endpoints and the web console."""

from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from medmlops import __version__, tracking
from medmlops.config import PROJECT_ROOT, Env

from .jobs import shutdown_manager
from .metrics import prometheus_middleware, refresh_model_gauges
from .routers import experiments, inference, monitoring, pipeline, system
from .services import build_state, load_settings

logger = logging.getLogger("medmlops.api")

_PUBLIC_API = {"/api/health", "/api/ready"}
_MAX_BODY_SLACK_MB = 2


class SPAStaticFiles(StaticFiles):
    """Serve the built React console and fall back to index.html."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except Exception:
            index = Path(self.directory) / "index.html"
            if index.exists() and "." not in Path(path).name:
                return await super().get_response("index.html", scope)
            raise


def _secure(response: Response) -> Response:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response


def _supplied_key(request: Request) -> str:
    key = (request.headers.get("x-api-key") or "").strip()
    if key:
        return key
    authorization = request.headers.get("authorization") or ""
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def create_app() -> FastAPI:
    settings = load_settings()
    state = build_state()

    try:
        tracking.ensure_tracking()
    except Exception:  # pragma: no cover - tracking optional for inference
        pass

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        refresh_model_gauges(state.predictor)
        env = Env()
        if env.get_bool("MEDMLOPS_BOOTSTRAP_FROM_REGISTRY") and not state.predictor.bundles:
            try:
                result = tracking.sync_champions(state.settings.model_dir)
                state.predictor.refresh()
                refresh_model_gauges(state.predictor)
                logger.info(
                    "registry bootstrap: %s champion(s) downloaded, %s error(s)",
                    len(result["downloaded"]),
                    len(result["errors"]),
                )
            except Exception as exc:  # pragma: no cover - deployment best effort
                logger.warning("registry bootstrap failed: %s", exc)
        try:
            yield
        finally:
            state.store.close()
            shutdown_manager()

    app = FastAPI(
        title="MedMLOps Console API",
        version=__version__,
        description=(
            "Medical image analysis platform: U-Net lesion segmentation, DenseNet "
            "diagnosis, MLflow experiment tracking, model registry and drift monitoring."
        ),
        lifespan=lifespan,
    )
    app.state.services = state

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.middleware("http")(prometheus_middleware)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        """Optional API-key auth, request-size guard and security headers."""
        path = request.url.path
        if settings.api_key and path.startswith("/api") and path not in _PUBLIC_API:
            if request.method != "OPTIONS" and not hmac.compare_digest(_supplied_key(request), settings.api_key):
                return _secure(
                    JSONResponse(
                        status_code=401,
                        content={"detail": "invalid or missing API key"},
                        headers={"WWW-Authenticate": "X-API-Key"},
                    )
                )
        if request.method in {"POST", "PUT", "PATCH"} and path.startswith(("/api/predict", "/api/feedback")):
            limit = (settings.max_upload_mb + _MAX_BODY_SLACK_MB) * 1024 * 1024
            try:
                declared = int(request.headers.get("content-length") or 0)
            except ValueError:
                declared = 0
            if declared > limit:
                return _secure(
                    JSONResponse(
                        status_code=413,
                        content={"detail": f"request body exceeds {settings.max_upload_mb} MB limit"},
                    )
                )
        return _secure(await call_next(request))

    for router in (system.router, inference.router, experiments.router, monitoring.router, pipeline.router):
        app.include_router(router)

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    dist = PROJECT_ROOT / "web" / "dist"
    if dist.exists():
        app.mount("/", SPAStaticFiles(directory=str(dist), html=True), name="console")

    return app


app = create_app()
