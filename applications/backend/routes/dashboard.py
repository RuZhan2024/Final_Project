from __future__ import annotations

"""Dashboard summary routes.

The dashboard page expects a resilient summary endpoint, so this route layer
preserves a full fallback payload even when DB-backed snapshots fail. Query and
schema logic lives in neighbouring services/repositories.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from ..db import get_conn
from ..db_schema import col_exists, table_exists
from ..repositories.residents_repository import one_resident_id, resident_exists
from ..runtime_state import get_last_pred_latency_ms
from ..services.dashboard_service import DashboardDeps, base_dashboard_summary, build_dashboard_summary_response


router = APIRouter()


def _dashboard_deps() -> DashboardDeps:
    """Bundle repository/schema helpers for the dashboard service boundary."""
    return DashboardDeps(
        resident_exists=resident_exists,
        one_resident_id=one_resident_id,
        table_exists=table_exists,
        col_exists=col_exists,
    )


@router.get("/api/dashboard/summary")
@router.get("/api/v1/dashboard/summary")
def dashboard_summary(resident_id: Optional[int] = None) -> Dict[str, Any]:
    """Return the dashboard summary without surfacing storage failures as 500s.

    The page uses this endpoint for top-level system health, so degraded DB
    reads are reflected in the payload instead of failing the whole request.
    """
    summary = base_dashboard_summary(get_last_pred_latency_ms() or 0)

    try:
        with get_conn() as conn:
            return build_dashboard_summary_response(
                conn,
                resident_id=resident_id,
                deps=_dashboard_deps(),
                last_pred_latency_ms=get_last_pred_latency_ms() or 0,
            )
    except (MySQLError, RuntimeError, TypeError, ValueError) as err:
        # Keep the response shape intact during DB outages so the frontend can
        # render a degraded state instead of handling a transport failure.
        summary["system"]["api_online"] = False
        summary["system"]["error"] = str(err)
        return summary


@router.get("/api/summary")
@router.get("/api/v1/summary")
def api_summary(resident_id: Optional[int] = None) -> Dict[str, Any]:
    """Legacy alias for the dashboard summary contract."""
    return dashboard_summary(resident_id=resident_id)
