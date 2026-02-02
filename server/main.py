# server/main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.caregivers import router as caregivers_router
from .routes.dashboard import router as dashboard_router
from .routes.events import router as events_router
from .routes.health import router as health_router
from .routes.monitor import router as monitor_router
from .routes.operating_points import router as operating_points_router
from .routes.settings import router as settings_router
from .routes.specs import router as specs_router


_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def create_app() -> FastAPI:
    app = FastAPI(title="Elder Fall Monitor API", version="0.3")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Core endpoints
    app.include_router(health_router)
    app.include_router(specs_router)
    app.include_router(operating_points_router)
    app.include_router(settings_router)
    app.include_router(events_router)
    app.include_router(dashboard_router)
    app.include_router(monitor_router)
    app.include_router(caregivers_router)

    return app


app = create_app()
