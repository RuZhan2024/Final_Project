from __future__ import annotations

"""Repository helpers for monitor runtime defaults and persisted monitor events."""

import json

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MonitorRuntimeDefaults:
    """Resident defaults loaded from DB rows for monitor runtime use.

    The object intentionally flattens system-settings and first-caregiver data
    into one repository result so service code can resolve runtime fallbacks
    without re-querying multiple tables.
    """

    dataset_code: Optional[str] = None
    use_mc: Optional[bool] = None
    mc_M: Optional[int] = None
    active_model_code: Optional[str] = None
    notify_on_every_fall: Optional[bool] = None
    notify_sms: Optional[bool] = None
    notify_phone: Optional[bool] = None
    op_code: Optional[str] = None
    caregiver_name: str = ""
    caregiver_email: str = ""
    caregiver_phone: str = ""
    caregiver_telegram_chat_id: str = ""


def load_monitor_runtime_defaults(
    conn: Any,
    *,
    resident_id: int,
    ensure_system_settings_schema,
    detect_variants,
    table_exists,
) -> MonitorRuntimeDefaults:
    """Load resident monitor defaults from system settings and caregiver rows.

    System settings are optional; caregiver fields are returned even when no
    settings row exists so notification-capable callers can still build a full
    runtime context.
    """

    ensure_system_settings_schema(conn)

    sys_row = None
    with conn.cursor() as cur:
        if table_exists(conn, "system_settings"):
            # Settings rows are optional in some demo/test deployments, so the
            # repository must tolerate the table existing but containing no row.
            cur.execute("SELECT * FROM system_settings WHERE resident_id=%s LIMIT 1", (resident_id,))
            sys_row = cur.fetchone()

    caregiver_name = ""
    caregiver_email = ""
    caregiver_phone = ""
    caregiver_telegram_chat_id = ""
    if table_exists(conn, "caregivers"):
        with conn.cursor() as cur:
            # The runtime path only needs one reachable caregiver profile for
            # default notifications, so it intentionally selects the first row.
            cur.execute(
                "SELECT * FROM caregivers WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
                (resident_id,),
            )
            caregiver_row = cur.fetchone() or {}
        if isinstance(caregiver_row, dict):
            caregiver_name = str(caregiver_row.get("name") or "").strip()
            caregiver_email = str(caregiver_row.get("email") or "").strip()
            caregiver_phone = str(caregiver_row.get("phone") or "").strip()
            caregiver_telegram_chat_id = str(caregiver_row.get("telegram_chat_id") or "").strip()

    if not isinstance(sys_row, dict):
        return MonitorRuntimeDefaults(
            caregiver_name=caregiver_name,
            caregiver_email=caregiver_email,
            caregiver_phone=caregiver_phone,
            caregiver_telegram_chat_id=caregiver_telegram_chat_id,
        )

    return MonitorRuntimeDefaults(
        dataset_code=sys_row.get("active_dataset_code"),
        use_mc=sys_row.get("mc_enabled"),
        mc_M=sys_row.get("mc_M"),
        active_model_code=sys_row.get("active_model_code"),
        notify_on_every_fall=sys_row.get("notify_on_every_fall"),
        notify_sms=sys_row.get("notify_sms"),
        notify_phone=sys_row.get("notify_phone"),
        op_code=sys_row.get("active_op_code"),
        caregiver_name=caregiver_name,
        caregiver_email=caregiver_email,
        caregiver_phone=caregiver_phone,
        caregiver_telegram_chat_id=caregiver_telegram_chat_id,
    )


def insert_monitor_event(
    conn: Any,
    *,
    resident_id: int,
    event_type: str,
    severity: str,
    model_code: str,
    score: float,
    meta: Dict[str, Any],
    table_exists,
) -> Optional[int]:
    """Insert a monitor-created event row into the active events schema.

    The inserted payload targets the common subset shared across deployed
    schemas. Optional fields such as ``status`` are written only when present.
    """

    if not table_exists(conn, "events"):
        return None

    cols = _event_columns(conn)
    insert_cols = ["resident_id", "type", "severity", "model_code", "operating_point_id", "score", "meta"]
    insert_vals = [resident_id, str(event_type), str(severity), str(model_code), None, float(score), json.dumps(meta)]
    if "status" in cols:
        # Newer schemas track review state directly on events. Keep runtime-created
        # rows aligned with the same pending-review contract used by the UI.
        insert_cols.append("status")
        insert_vals.append("pending_review")

    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO events ({', '.join(insert_cols)}) VALUES ({', '.join(['%s'] * len(insert_cols))})",
            tuple(insert_vals),
        )
        return cur.lastrowid


def _event_columns(conn: Any) -> set[str]:
    """Return event column names across MySQL and SQLite backends."""

    with conn.cursor() as cur:
        backend = str(getattr(conn, "db_backend", "mysql")).lower()
        if backend == "sqlite":
            # SQLite uses PRAGMA metadata instead of SHOW COLUMNS, so repository
            # code normalizes the result into one common set-of-names contract.
            cur.execute("PRAGMA table_info(`events`)")
            rows = cur.fetchall() or []
            return {str(r.get("name")) for r in rows if isinstance(r, dict) and r.get("name") is not None}
        cur.execute("SHOW COLUMNS FROM `events`")
        rows = cur.fetchall() or []
        return {str(r.get("Field")) for r in rows if isinstance(r, dict) and r.get("Field") is not None}
