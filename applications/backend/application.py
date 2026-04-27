from __future__ import annotations

"""FastAPI application factory and route registration."""

import logging
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from .config import AppConfig, get_app_config

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Install a basic root logger once for local app startup."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _register_routes(app: FastAPI, *, include_runtime_routes: bool = True) -> None:
    """Register product-surface routes, with monitor routes optionally disabled for tests."""
    from .routes.caregivers import router as caregivers_router
    from .routes.dashboard import router as dashboard_router
    from .routes.events import router as events_router
    from .routes.health import router as health_router
    from .routes.notifications import router as notifications_router
    from .routes.operating_points import router as operating_points_router
    from .routes.settings import router as settings_router
    from .routes.specs import router as specs_router
    from .routes.twilio_webhook import router as twilio_webhook_router

    app.include_router(health_router)
    app.include_router(specs_router)
    app.include_router(operating_points_router)
    app.include_router(settings_router)
    app.include_router(events_router)
    app.include_router(dashboard_router)
    app.include_router(notifications_router)
    app.include_router(caregivers_router)
    app.include_router(twilio_webhook_router)
    if include_runtime_routes:
        from .routes.monitor import router as monitor_router

        app.include_router(monitor_router)


def create_app(config: AppConfig | None = None, *, include_runtime_routes: bool = True) -> FastAPI:
    """Create the configured FastAPI app instance used by ASGI entrypoints and tests."""
    _configure_logging()
    cfg = config or get_app_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "app_startup db_backend=%s sqlite_path=%s event_clips_dir=%s app_base_url=%s runtime_routes=%s",
            cfg.db_backend,
            cfg.sqlite_path,
            cfg.event_clips_dir,
            cfg.app_base_url,
            include_runtime_routes,
        )
        yield

    app = FastAPI(title="Elder Fall Monitor API", version="0.3", lifespan=lifespan)
    app.state.app_config = cfg

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        # Keep request logging centralized here so route modules stay transport-focused.
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "http_request method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    _register_routes(app, include_runtime_routes=include_runtime_routes)
    return app
