from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import AppConfig, get_app_config


def _register_routes(app: FastAPI, *, include_runtime_routes: bool = True) -> None:
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
    cfg = config or get_app_config()
    app = FastAPI(title="Elder Fall Monitor API", version="0.3")
    app.state.app_config = cfg

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_routes(app, include_runtime_routes=include_runtime_routes)
    return app
