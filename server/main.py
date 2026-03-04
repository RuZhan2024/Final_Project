# server/main.py
from __future__ import annotations

import logging
import os
import time
from collections import deque

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from .routes.caregivers import router as caregivers_router
from .routes.dashboard import router as dashboard_router
from .routes.events import router as events_router
from .routes.health import router as health_router
from .routes.monitor import router as monitor_router
from .routes.notifications import router as notifications_router
from .routes.operating_points import router as operating_points_router
from .routes.settings import router as settings_router
from .routes.specs import router as specs_router


_DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def _compute_allowed_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return list(_DEFAULT_ALLOWED_ORIGINS)
    out = [o.strip() for o in raw.split(",") if o.strip()]
    return out or list(_DEFAULT_ALLOWED_ORIGINS)


_ALLOWED_ORIGINS = _compute_allowed_origins()
_LATENCY_WINDOW = int(os.getenv("MONITOR_LATENCY_WINDOW", "200"))
_LATENCY_LOG_EVERY = int(os.getenv("MONITOR_LATENCY_LOG_EVERY", "50"))
_monitor_lat_ms: deque[float] = deque(maxlen=max(10, _LATENCY_WINDOW))
_monitor_req_count = 0
logger = logging.getLogger(__name__)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round((p / 100.0) * (len(vals) - 1)))
    idx = max(0, min(idx, len(vals) - 1))
    return float(vals[idx])


def create_app() -> FastAPI:
    app = FastAPI(title="Elder Fall Monitor API", version="0.3")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def monitor_latency_middleware(request: Request, call_next):
        global _monitor_req_count
        path = request.url.path
        should_track = path.endswith("/api/monitor/predict_window") or path.endswith("/api/v1/monitor/predict_window")
        if not should_track:
            return await call_next(request)

        t0 = time.perf_counter()
        response = await call_next(request)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        _monitor_lat_ms.append(float(dt_ms))
        _monitor_req_count += 1
        if _monitor_req_count % max(1, _LATENCY_LOG_EVERY) == 0:
            vals = list(_monitor_lat_ms)
            logger.info(
                "monitor.latency rolling_ms count=%d window=%d p50=%.2f p95=%.2f min=%.2f max=%.2f",
                _monitor_req_count,
                len(vals),
                _percentile(vals, 50.0),
                _percentile(vals, 95.0),
                min(vals) if vals else 0.0,
                max(vals) if vals else 0.0,
            )
        return response

    # Core endpoints
    app.include_router(health_router)
    app.include_router(specs_router)
    app.include_router(operating_points_router)
    app.include_router(settings_router)
    app.include_router(events_router)
    app.include_router(dashboard_router)
    app.include_router(monitor_router)
    app.include_router(notifications_router)
    app.include_router(caregivers_router)

    return app


app = create_app()
