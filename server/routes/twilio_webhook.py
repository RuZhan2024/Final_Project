from __future__ import annotations

import json
import re

from typing import Optional
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from ..db import get_conn_optional
from ..db_schema import has_col
from ..deploy_ops import detect_variants
from ..notifications import get_notification_manager


router = APIRouter()
_detect_variants = detect_variants
_has_col = has_col
_EVENT_REF_RE = re.compile(r"\bref[:#\s-]*([A-Za-z0-9_-]+)\b", re.IGNORECASE)


def _map_reply_code(reply_code: str) -> tuple[str, str]:
    code = str(reply_code or "").strip()
    if code == "1":
        return "false_alarm", "false_positive"
    if code == "2":
        return "confirmed_fall", "confirmed_fall"
    if code == "3":
        return "dismissed", "assistance_provided"
    raise HTTPException(status_code=400, detail="Unsupported reply code")


def _extract_event_id(body: str, explicit_event_id: Optional[str]) -> Optional[str]:
    if explicit_event_id:
        return str(explicit_event_id).strip() or None
    match = _EVENT_REF_RE.search(str(body or ""))
    if match:
        return match.group(1)
    return None


def _update_canonical_event_status(event_id: str, db_status: str) -> bool:
    with get_conn_optional() as conn:
        if conn is None:
            return False
        variants = detect_variants(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM events WHERE id=%s LIMIT 1", (event_id,))
            row = cur.fetchone()
            if not row:
                return False
            if variants.get("events") == "v2" and has_col(conn, "events", "status"):
                cur.execute("UPDATE events SET status=%s WHERE id=%s", (db_status, event_id))
            elif has_col(conn, "events", "meta"):
                cur.execute("SELECT meta FROM events WHERE id=%s LIMIT 1", (event_id,))
                meta_row = cur.fetchone() or {}
                meta = meta_row.get("meta")
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except (TypeError, json.JSONDecodeError):
                        meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                meta["status"] = db_status
                cur.execute("UPDATE events SET meta=%s WHERE id=%s", (json.dumps(meta), event_id))
            else:
                return False
        conn.commit()
        return True


@router.post("/twilio/webhook")
async def twilio_webhook(request: Request):
    body_bytes = await request.body()
    form = parse_qs(body_bytes.decode("utf-8", errors="replace"), keep_blank_values=True)
    reply_body = str((form.get("Body") or [""])[0]).strip()
    resident_id = int(str((form.get("resident_id") or ["1"])[0]).strip() or "1")
    explicit_event_id = (form.get("event_id") or [None])[0]

    db_status, feedback_value = _map_reply_code(reply_body)
    event_id = _extract_event_id(reply_body, explicit_event_id)

    manager = get_notification_manager()
    if not event_id:
        event_id = manager.store.get_most_recent_unresolved_event_id(resident_id)
    if not event_id:
        raise HTTPException(status_code=404, detail="No unresolved alert event found")

    updated = _update_canonical_event_status(event_id, db_status)
    manager.store.mark_feedback(event_id, resident_id, reply_body, feedback_value)
    if feedback_value == "assistance_provided":
        manager.store.resolve_event(event_id)

    return {
        "ok": True,
        "event_id": event_id,
        "reply_code": reply_body,
        "db_status": db_status,
        "feedback": feedback_value,
        "canonical_updated": bool(updated),
    }
