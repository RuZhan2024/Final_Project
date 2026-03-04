from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, Query

from ..db import get_conn_optional

router = APIRouter()


@router.get("/api/notifications")
@router.get("/api/v1/notifications")
def list_notifications(
    resident_id: int = Query(1, description="Resident ID"),
    limit: int = Query(50, ge=1, le=500),
) -> Dict[str, Any]:
    with get_conn_optional() as conn:
        if conn is None:
            return {"resident_id": resident_id, "rows": [], "db_available": False}
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, resident_id, ts, channel, status, message, event_id "
                    "FROM notifications_log WHERE resident_id=%s ORDER BY id DESC LIMIT %s",
                    (int(resident_id), int(limit)),
                )
                rows = cur.fetchall() or []
            return {"resident_id": int(resident_id), "rows": rows, "db_available": True}
        except Exception as e:
            return {"resident_id": int(resident_id), "rows": [], "db_available": False, "error": str(e)}


@router.post("/api/notifications/test")
@router.post("/api/v1/notifications/test")
def test_notification(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Minimal endpoint used by the monitor UI fallback test action."""
    p = payload if isinstance(payload, dict) else {}
    resident_id = int(p.get("resident_id") or 1)
    channel = str(p.get("channel") or "test")
    message = str(p.get("message") or "Manual test notification")

    with get_conn_optional() as conn:
        if conn is None:
            return {"ok": True, "accepted": True, "persisted": False, "reason": "db_unavailable", "payload": p}
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO notifications_log (resident_id, channel, status, message) VALUES (%s,%s,%s,%s)",
                    (resident_id, channel, "queued", message),
                )
                row_id = cur.lastrowid
            conn.commit()
            return {"ok": True, "accepted": True, "persisted": True, "id": row_id, "payload": p}
        except Exception as e:
            return {"ok": True, "accepted": True, "persisted": False, "error": str(e), "payload": p}
