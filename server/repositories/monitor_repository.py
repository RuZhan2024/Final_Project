from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MonitorRuntimeDefaults:
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
    ensure_system_settings_schema(conn)
    variants = detect_variants(conn)

    sys_row = None
    with conn.cursor() as cur:
        if variants.get("settings") == "v2" and table_exists(conn, "system_settings"):
            cur.execute("SELECT * FROM system_settings WHERE resident_id=%s LIMIT 1", (resident_id,))
            sys_row = cur.fetchone()
        elif table_exists(conn, "settings"):
            cur.execute("SELECT * FROM settings WHERE resident_id=%s LIMIT 1", (resident_id,))
            sys_row = cur.fetchone()

    caregiver_name = ""
    caregiver_email = ""
    caregiver_phone = ""
    caregiver_telegram_chat_id = ""
    if table_exists(conn, "caregivers"):
        with conn.cursor() as cur:
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
    if not table_exists(conn, "events"):
        return None

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO events (resident_id, type, severity, model_code, operating_point_id, score, meta) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (
                resident_id,
                str(event_type),
                str(severity),
                str(model_code),
                None,
                float(score),
                json.dumps(meta),
            ),
        )
        return cur.lastrowid
