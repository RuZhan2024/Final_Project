from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from .. import core
from ..core import _col_exists, _one_resident_id, _resident_exists, _table_exists
from ..db import get_conn
from ..services.dashboard_service import DashboardDeps, base_dashboard_summary, build_dashboard_summary_response


router = APIRouter()


def _dashboard_deps() -> DashboardDeps:
    return DashboardDeps(
        resident_exists=_resident_exists,
        one_resident_id=_one_resident_id,
        table_exists=_table_exists,
        col_exists=_col_exists,
    )


@router.get("/api/dashboard/summary")
@router.get("/api/v1/dashboard/summary")
def dashboard_summary(resident_id: Optional[int] = None) -> Dict[str, Any]:
    """Summary used by /Dashboard. Never returns 500."""
    summary = base_dashboard_summary(core.LAST_PRED_LATENCY_MS or 0)

    try:
        with get_conn() as conn:
            return build_dashboard_summary_response(
                conn,
                resident_id=resident_id,
                deps=_dashboard_deps(),
                last_pred_latency_ms=core.LAST_PRED_LATENCY_MS or 0,
            )
    except (MySQLError, RuntimeError, TypeError, ValueError) as err:
        summary["system"]["api_online"] = False
        summary["system"]["error"] = str(err)
        return summary


@router.get("/api/summary")
@router.get("/api/v1/summary")
def api_summary(resident_id: Optional[int] = None) -> Dict[str, Any]:
    return dashboard_summary(resident_id=resident_id)
