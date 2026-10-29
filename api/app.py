"""FastAPI application: inference API, MLOps endpoints and the web console."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from medmlops import __version__
from medmlops.config import PROJECT_ROOT

from .metrics import prometheus_middleware, refresh_model_gauges
from .routers import experiments, inference, monitoring, pipeline, system
from .services import build_state, load_settings


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


def create_app() -> FastAPI:
    settings = load_settings()
    state = build_state()

    from medmlops import tracking

    try:
        tracking.ensure_tracking()
    except Exception:  # pragma: no cover - tracking optional for inference
        pass

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        refresh_model_gauges(state.predictor)
        yield

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
    app.middleware("http")(prometheus_middleware)

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
