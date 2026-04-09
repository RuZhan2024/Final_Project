from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Body, Query

from ..core import _col_exists, _ensure_caregivers_table
from ..db import get_conn_optional
from ..notifications import get_notification_manager
from ..notifications.models import NotificationPreferences, SafeGuardEvent

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
    """Trigger a real Safe Guard test notification when possible."""
    p = payload if isinstance(payload, dict) else {}
    resident_id = int(p.get("resident_id") or 1)
    channel = str(p.get("channel") or "telegram")
    message = str(p.get("message") or "Manual Telegram test notification")
    manager = get_notification_manager()
    caregiver_name = ""
    caregiver_chat_id = ""
    row_id = None

    with get_conn_optional() as conn:
        if conn is not None:
            with conn.cursor() as cur:
                try:
                    _ensure_caregivers_table(conn)
                    select_cols = "name"
                    if _col_exists(conn, "caregivers", "telegram_chat_id"):
                        select_cols += ", telegram_chat_id"
                    cur.execute(
                        f"SELECT {select_cols} FROM caregivers WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
                        (resident_id,),
                    )
                    cg = cur.fetchone() or {}
                    caregiver_name = str(cg.get("name") or "").strip()
                    caregiver_chat_id = str(cg.get("telegram_chat_id") or "").strip()
                except Exception:
                    caregiver_name = ""
                    caregiver_chat_id = ""

                try:
                    cur.execute(
                        "INSERT INTO notifications_log (resident_id, channel, status, message) VALUES (%s,%s,%s,%s)",
                        (resident_id, channel, "queued", message),
                    )
                    row_id = cur.lastrowid
                    conn.commit()
                except Exception:
                    row_id = None

    manager.handle_event(
        SafeGuardEvent(
            event_id=f"manual-test-{int(datetime.now(timezone.utc).timestamp())}",
            resident_id=resident_id,
            location="manual_test_notification",
            probability=0.99,
            uncertainty=0.01,
            threshold=0.71,
            margin=0.28,
            triage_state="fall",
            safe_alert=True,
            recall_alert=True,
            model_code="TCN",
            dataset_code="caucafall",
            op_code="OP-2",
            timestamp=datetime.now(timezone.utc),
            source="notifications.test",
            notes=message,
        ),
        NotificationPreferences(
            telegram_enabled=True,
            caregiver_name=caregiver_name,
            caregiver_telegram_chat_id=caregiver_chat_id,
        ),
    )
    return {
        "ok": True,
        "accepted": True,
        "persisted": bool(row_id),
        "id": row_id,
        "safe_guard_enabled": bool(manager.enabled),
        "channel": channel,
        "payload": p,
    }
