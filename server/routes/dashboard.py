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


router = APIRouter()


@router.get("/api/dashboard/summary")
@router.get("/api/v1/dashboard/summary")
def dashboard_summary(resident_id: Optional[int] = None) -> Dict[str, Any]:
    """Summary used by /Dashboard. Never returns 500."""

    summary: Dict[str, Any] = {
        "status": "normal",
        "today": {"falls_detected": 0, "false_alarms": 0},
        "system": {
            "model_name": "HYBRID",
            "monitoring_enabled": False,
            "last_latency_ms": int(core.LAST_PRED_LATENCY_MS or 0),
            "api_online": True,
        },
    }

    try:
        with get_conn() as conn:
            rid = resident_id if resident_id and _resident_exists(conn, int(resident_id)) else _one_resident_id(conn)

            # settings / model
            if _table_exists(conn, "system_settings"):
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM system_settings WHERE resident_id=%s ORDER BY id ASC LIMIT 1", (rid,))
                    sys_row = cur.fetchone() or {}
                    if isinstance(sys_row, dict) and "monitoring_enabled" in sys_row:
                        summary["system"]["monitoring_enabled"] = bool(sys_row.get("monitoring_enabled", 1))
                    if isinstance(sys_row, dict) and sys_row.get("active_model_code"):
                        summary["system"]["model_name"] = str(sys_row.get("active_model_code") or summary["system"]["model_name"])

                    active_model_id = sys_row.get("active_model_id") if isinstance(sys_row, dict) else None
                    if active_model_id and _table_exists(conn, "models"):
                        cur.execute("SELECT * FROM models WHERE id=%s LIMIT 1", (active_model_id,))
                        mrow = cur.fetchone() or {}
                        if isinstance(mrow, dict):
                            summary["system"]["model_name"] = (
                                mrow.get("name")
                                or mrow.get("model_code")
                                or mrow.get("code")
                                or summary["system"]["model_name"]
                            )
            elif _table_exists(conn, "settings") and _col_exists(conn, "settings", "monitoring_enabled"):
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM settings WHERE resident_id=%s LIMIT 1", (rid,))
                    row = cur.fetchone() or {}
                    if isinstance(row, dict):
                        summary["system"]["monitoring_enabled"] = bool(row.get("monitoring_enabled", 1))
                        summary["system"]["model_name"] = row.get("active_model_code") or summary["system"]["model_name"]

            # counts today
            today_falls = 0
            today_false = 0
            with conn.cursor() as cur:
                if _table_exists(conn, "events") and _col_exists(conn, "events", "event_type"):
                    cur.execute(
                        "SELECT COUNT(*) AS c FROM events "
                        "WHERE DATE(created_at)=CURDATE() AND UPPER(event_type) IN ('FALL','FALL_DETECTED','FALL_CONFIRMED')"
                    )
                    r = cur.fetchone() or {}
                    today_falls = int(r.get("c", 0)) if isinstance(r, dict) else int(list(r)[0])
                    cur.execute(
                        "SELECT COUNT(*) AS c FROM events "
                        "WHERE DATE(created_at)=CURDATE() AND UPPER(event_type) IN ('FALSE_ALARM','FALSE','FALSE_POSITIVE')"
                    )
                    r = cur.fetchone() or {}
                    today_false = int(r.get("c", 0)) if isinstance(r, dict) else int(list(r)[0])
                elif _table_exists(conn, "fall_events"):
                    cur.execute(
                        "SELECT "
                        "SUM(CASE WHEN event_type='fall_detected' THEN 1 ELSE 0 END) AS falls_detected, "
                        "SUM(CASE WHEN event_type='false_alarm' THEN 1 ELSE 0 END) AS false_alarms "
                        "FROM fall_events WHERE DATE(created_at)=CURDATE()"
                    )
                    r = cur.fetchone() or {}
                    if isinstance(r, dict):
                        today_falls = int(r.get("falls_detected") or 0)
                        today_false = int(r.get("false_alarms") or 0)

            summary["today"]["falls_detected"] = today_falls
            summary["today"]["false_alarms"] = today_false
            summary["status"] = "alert" if today_falls > 0 else "normal"

            # heartbeat latency if available
            with conn.cursor() as cur:
                if _table_exists(conn, "heartbeat") and _col_exists(conn, "heartbeat", "latency_ms"):
                    cur.execute("SELECT latency_ms FROM heartbeat ORDER BY created_at DESC LIMIT 1")
                    r = cur.fetchone()
                    if isinstance(r, dict) and r.get("latency_ms") is not None:
                        summary["system"]["last_latency_ms"] = int(r["latency_ms"])

            return summary

    except (MySQLError, RuntimeError, TypeError, ValueError) as e:
        summary["system"]["api_online"] = False
        summary["system"]["error"] = str(e)
        return summary


@router.get("/api/summary")
@router.get("/api/v1/summary")
def api_summary(resident_id: Optional[int] = None) -> Dict[str, Any]:
    """Alias for /api/dashboard/summary (for Monitor UI)."""
    return dashboard_summary(resident_id=resident_id)
