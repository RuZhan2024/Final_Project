from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..core import normalize_dataset_code
from .monitor_runtime_service import MonitorRuntimeContext
from .value_coercion import coerce_bool


def resolve_mc_runtime(
    *,
    requested_use_mc: Any,
    requested_mc_m: Any,
    sys_row: Optional[Dict[str, Any]],
    is_replay: bool,
) -> Tuple[bool, int, bool, int]:
    use_mc = requested_use_mc
    mc_m = requested_mc_m

    if isinstance(sys_row, dict):
        if use_mc is None and sys_row.get("mc_enabled") is not None:
            use_mc = coerce_bool(sys_row.get("mc_enabled"), False)
        if mc_m is None and sys_row.get("mc_M") is not None:
            try:
                mc_m = int(sys_row.get("mc_M"))
            except (TypeError, ValueError):
                mc_m = None

    requested_use_mc_bool = coerce_bool(use_mc, False)
    try:
        requested_mc_m_int = max(1, int(mc_m)) if mc_m is not None else 10
    except (TypeError, ValueError):
        requested_mc_m_int = 10

    effective_use_mc = requested_use_mc_bool and (not is_replay)
    effective_mc_m = requested_mc_m_int if effective_use_mc else 1
    return requested_use_mc_bool, requested_mc_m_int, effective_use_mc, effective_mc_m


@dataclass(frozen=True)
class MonitorRequestContext:
    resident_id: int
    event_location: str
    dataset_code: str
    op_code: str
    active_model_code: str
    cooldown_sec: int
    requested_use_mc: bool
    requested_mc_M: int
    effective_use_mc: bool
    effective_mc_M: int
    runtime: MonitorRuntimeContext


def load_monitor_request_context(
    *,
    conn: Any,
    resident_id: int,
    input_source: str,
    mode: str,
    dataset_code: Optional[str],
    op_code: Optional[str],
    event_location: Optional[str],
    active_model_code: str,
    requested_use_mc: Any,
    requested_mc_M: Any,
    is_replay: bool,
    ensure_system_settings_schema,
    detect_variants,
    table_exists,
) -> MonitorRequestContext:
    sys_row = None
    resolved_dataset_code = normalize_dataset_code(dataset_code)
    resolved_op_code = str(op_code or "").upper().strip()
    resolved_active_model = str(active_model_code or mode.upper())
    cooldown_sec = 30
    notify_on_every_fall = True
    notify_sms = False
    notify_phone = False
    caregiver_name = ""
    caregiver_email = ""
    caregiver_phone = ""

    if conn is not None:
        ensure_system_settings_schema(conn)
        variants = detect_variants(conn)

        with conn.cursor() as cur:
            if variants.get("settings") == "v2" and table_exists(conn, "system_settings"):
                cur.execute("SELECT * FROM system_settings WHERE resident_id=%s LIMIT 1", (resident_id,))
                sys_row = cur.fetchone()
            elif table_exists(conn, "settings"):
                cur.execute("SELECT * FROM settings WHERE resident_id=%s LIMIT 1", (resident_id,))
                sys_row = cur.fetchone()

        if isinstance(sys_row, dict):
            if not resolved_dataset_code and sys_row.get("active_dataset_code"):
                resolved_dataset_code = normalize_dataset_code(sys_row.get("active_dataset_code"))
            if sys_row.get("alert_cooldown_sec") is not None:
                cooldown_sec = int(sys_row.get("alert_cooldown_sec"))
            if sys_row.get("active_model_code"):
                resolved_active_model = str(sys_row.get("active_model_code") or resolved_active_model)
            if sys_row.get("notify_on_every_fall") is not None:
                notify_on_every_fall = coerce_bool(sys_row.get("notify_on_every_fall"), True)
            if sys_row.get("notify_sms") is not None:
                notify_sms = coerce_bool(sys_row.get("notify_sms"), False)
            if sys_row.get("notify_phone") is not None:
                notify_phone = coerce_bool(sys_row.get("notify_phone"), False)

            if (not resolved_op_code) and sys_row.get("active_op_code"):
                resolved_op_code = str(sys_row.get("active_op_code") or "").upper().strip()

            op_id = None
            for key in ("active_operating_point", "active_operating_point_id"):
                if sys_row.get(key) is not None:
                    op_id = sys_row.get(key)
                    break
            if (not resolved_op_code) and op_id and table_exists(conn, "operating_points"):
                with conn.cursor() as cur:
                    cur.execute("SELECT code FROM operating_points WHERE id=%s LIMIT 1", (int(op_id),))
                    row = cur.fetchone() or {}
                    if isinstance(row, dict) and row.get("code"):
                        resolved_op_code = str(row.get("code") or "").upper()

        if table_exists(conn, "caregivers"):
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT name, email, phone FROM caregivers WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
                    (resident_id,),
                )
                caregiver_row = cur.fetchone() or {}
            if isinstance(caregiver_row, dict):
                caregiver_name = str(caregiver_row.get("name") or "").strip()
                caregiver_email = str(caregiver_row.get("email") or "").strip()
                caregiver_phone = str(caregiver_row.get("phone") or "").strip()

    if not resolved_dataset_code:
        resolved_dataset_code = "le2i"
    if not resolved_op_code:
        resolved_op_code = "OP-2"

    requested_use_mc_bool, requested_mc_m_int, effective_use_mc, effective_mc_m = resolve_mc_runtime(
        requested_use_mc=requested_use_mc,
        requested_mc_m=requested_mc_M,
        sys_row=sys_row if isinstance(sys_row, dict) else None,
        is_replay=is_replay,
    )

    runtime = MonitorRuntimeContext(
        dataset_code=str(resolved_dataset_code),
        op_code=str(resolved_op_code),
        use_mc=bool(requested_use_mc_bool),
        mc_M=int(requested_mc_m_int),
        active_model_code=str(resolved_active_model),
        notify_on_every_fall=bool(notify_on_every_fall),
        notify_sms=bool(notify_sms),
        notify_phone=bool(notify_phone),
        caregiver_name=str(caregiver_name),
        caregiver_email=str(caregiver_email),
        caregiver_phone=str(caregiver_phone),
    )

    return MonitorRequestContext(
        resident_id=int(resident_id),
        event_location=str(event_location or input_source or "unknown").strip() or "unknown",
        dataset_code=str(resolved_dataset_code),
        op_code=str(resolved_op_code),
        active_model_code=str(resolved_active_model),
        cooldown_sec=int(cooldown_sec),
        requested_use_mc=bool(requested_use_mc_bool),
        requested_mc_M=int(requested_mc_m_int),
        effective_use_mc=bool(effective_use_mc),
        effective_mc_M=int(effective_mc_m),
        runtime=runtime,
    )
