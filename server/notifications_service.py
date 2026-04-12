from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .core import _col_exists, _ensure_caregivers_table, _table_exists
from .notifications import get_notification_manager


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    try:
        s = str(v).strip().lower()
    except Exception:
        return default
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    try:
        return bool(int(v))
    except Exception:
        return bool(v)


def dispatch_fall_notifications(
    conn: Any,
    *,
    resident_id: int,
    event_id: Optional[int],
    p_fall: Optional[float] = None,
    source: str = "monitor",
) -> Dict[str, Any]:
    """Create notification log rows for a fall event based on settings.

    This function is DB-only and best-effort. It never raises to caller.
    """
    out: Dict[str, Any] = {
        "attempted": False,
        "enabled": False,
        "channels": [],
        "rows_written": 0,
        "safe_guard_enabled": False,
    }
    try:
        out["safe_guard_enabled"] = bool(get_notification_manager().enabled)
        if not _table_exists(conn, "notifications_log"):
            out["reason"] = "notifications_log_missing"
            return out

        notify_on_every_fall = True
        with conn.cursor() as cur:
            if _table_exists(conn, "system_settings"):
                cur.execute(
                    "SELECT * FROM system_settings WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
                    (int(resident_id),),
                )
                s = cur.fetchone() or {}
                notify_on_every_fall = _as_bool(s.get("notify_on_every_fall"), True)

            if not notify_on_every_fall:
                out["enabled"] = False
                out["reason"] = "notify_disabled"
                return out
            out["enabled"] = True

            channels: List[str] = ["telegram"]
            out["channels"] = channels

            # Optional caregiver details for human-readable message.
            caregiver_name = None
            caregiver_telegram_chat_id = None
            if _table_exists(conn, "caregivers"):
                _ensure_caregivers_table(conn)
                select_cols = "name"
                if _col_exists(conn, "caregivers", "telegram_chat_id"):
                    select_cols += ", telegram_chat_id"
                cur.execute(
                    f"SELECT {select_cols} FROM caregivers WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
                    (int(resident_id),),
                )
                cg = cur.fetchone() or {}
                caregiver_name = cg.get("name")
                caregiver_telegram_chat_id = cg.get("telegram_chat_id")

            score_txt = f"{float(p_fall):.3f}" if p_fall is not None else "n/a"
            msg = (
                f"Fall detected (resident={resident_id}, event_id={event_id}, "
                f"p_fall={score_txt}, source={source}, ts={datetime.now(timezone.utc).isoformat()})"
            )
            if caregiver_name or caregiver_telegram_chat_id:
                msg += f" -> caregiver={caregiver_name or 'unknown'} telegram_chat_id={caregiver_telegram_chat_id or 'n/a'}"

            wrote = 0
            for ch in channels:
                cols = ["resident_id", "channel", "status", "message"]
                vals = [int(resident_id), ch, "queued", msg]
                if _col_exists(conn, "notifications_log", "event_id"):
                    cols.append("event_id")
                    vals.append(int(event_id) if event_id is not None else None)
                sql = f"INSERT INTO notifications_log ({', '.join(cols)}) VALUES ({', '.join(['%s']*len(cols))})"
                cur.execute(sql, tuple(vals))
                wrote += 1
            out["attempted"] = True
            out["rows_written"] = wrote

            # Best effort: mark events.alert_sent=1 when available.
            if event_id is not None and _table_exists(conn, "events") and _col_exists(conn, "events", "alert_sent"):
                try:
                    cur.execute("UPDATE events SET alert_sent=%s WHERE id=%s", (1, int(event_id)))
                except Exception:
                    pass

        return out
    except Exception as e:
        out["error"] = str(e)
        return out
