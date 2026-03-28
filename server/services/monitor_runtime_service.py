from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..notifications import NotificationPreferences, SafeGuardEvent, get_notification_manager
from ..notifications_service import dispatch_fall_notifications
from ..repositories.monitor_repository import MonitorRuntimeDefaults, insert_monitor_event


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"1", "true", "yes", "on"}:
            return True
        if stripped in {"0", "false", "no", "off"}:
            return False
    return bool(value)


@dataclass(frozen=True)
class MonitorRuntimeContext:
    dataset_code: str
    op_code: str
    use_mc: bool
    mc_M: int
    active_model_code: str
    notify_on_every_fall: bool
    notify_sms: bool
    notify_phone: bool
    caregiver_name: str
    caregiver_email: str
    caregiver_phone: str


def merge_monitor_runtime_defaults(
    *,
    dataset_code: str,
    op_code: str,
    use_mc: Any,
    mc_M: Any,
    active_model_code: str,
    defaults: MonitorRuntimeDefaults,
) -> MonitorRuntimeContext:
    merged_dataset = str(dataset_code or defaults.dataset_code or "caucafall")
    merged_op_code = str(op_code or defaults.op_code or "OP-2").upper().strip() or "OP-2"

    merged_use_mc = _coerce_optional_bool(use_mc)
    if merged_use_mc is None:
        merged_use_mc = _coerce_optional_bool(defaults.use_mc)
    if merged_use_mc is None:
        merged_use_mc = True

    merged_mc_M = int(mc_M if mc_M is not None else (defaults.mc_M if defaults.mc_M is not None else 10))
    merged_model = str(active_model_code or defaults.active_model_code or "TCN")
    notify_on_every_fall = _coerce_optional_bool(defaults.notify_on_every_fall)
    notify_sms = _coerce_optional_bool(defaults.notify_sms)
    notify_phone = _coerce_optional_bool(defaults.notify_phone)

    return MonitorRuntimeContext(
        dataset_code=merged_dataset,
        op_code=merged_op_code,
        use_mc=bool(merged_use_mc),
        mc_M=merged_mc_M,
        active_model_code=merged_model,
        notify_on_every_fall=True if notify_on_every_fall is None else bool(notify_on_every_fall),
        notify_sms=False if notify_sms is None else bool(notify_sms),
        notify_phone=False if notify_phone is None else bool(notify_phone),
        caregiver_name=defaults.caregiver_name,
        caregiver_email=defaults.caregiver_email,
        caregiver_phone=defaults.caregiver_phone,
    )


@dataclass(frozen=True)
class MonitorPersistencePlan:
    persist_event_type: Optional[str]
    persist_cd_key: str
    persist_dedup_key: str
    persist_in_cooldown: bool
    persist_suppressed: bool
    persist_dedup_hits: int
    cooldown_s: float
    last_persist_ts: float


def resolve_monitor_persistence_plan(
    *,
    session_state: Dict[str, Any],
    resident_id: int,
    dataset_code: str,
    mode: str,
    op_code: str,
    current_t_s: float,
    cooldown_s: float,
    is_replay: bool,
    persist: bool,
    triage_state: str,
    prev_triage_state: str,
    started_event: bool,
) -> MonitorPersistencePlan:
    persist_event_type: Optional[str] = None
    if is_replay:
        if triage_state in {"fall", "uncertain"} and prev_triage_state != str(triage_state):
            persist_event_type = str(triage_state)
    elif started_event and triage_state == "fall":
        persist_event_type = "fall"

    persist_cd_key = f"persist_last_ts:{resident_id}:{dataset_code}:{mode}:{op_code}:{persist_event_type or 'none'}"
    last_persist_ts = float(session_state.get(persist_cd_key, 0.0) or 0.0)
    persist_in_cooldown = bool(persist_event_type) and ((current_t_s - last_persist_ts) < float(cooldown_s))

    persist_dedup_key = (
        f"persist_dedup_hits:{resident_id}:{dataset_code}:{mode}:{op_code}:{persist_event_type or 'none'}"
    )
    persist_dedup_hits = int(session_state.get(persist_dedup_key, 0) or 0)
    persist_suppressed = bool(persist and persist_event_type and persist_in_cooldown)
    if persist_suppressed:
        persist_dedup_hits += 1
        session_state[persist_dedup_key] = persist_dedup_hits

    return MonitorPersistencePlan(
        persist_event_type=persist_event_type,
        persist_cd_key=persist_cd_key,
        persist_dedup_key=persist_dedup_key,
        persist_in_cooldown=bool(persist_in_cooldown),
        persist_suppressed=bool(persist_suppressed),
        persist_dedup_hits=persist_dedup_hits,
        cooldown_s=max(1.0, float(cooldown_s)),
        last_persist_ts=last_persist_ts,
    )


def persist_monitor_event(
    conn: Any,
    *,
    session_state: Dict[str, Any],
    persistence: MonitorPersistencePlan,
    current_t_s: float,
    resident_id: int,
    event_location: str,
    input_source: str,
    requested_mode: str,
    effective_mode: str,
    runtime: MonitorRuntimeContext,
    table_exists,
    triage_state: str,
    safe_alert: bool,
    safe_state_out: str,
    recall_alert: bool,
    recall_state_out: str,
    expected_fps: float,
    capture_fps_est: Optional[float],
    p_display: float,
    primary_probability: float,
    primary_uncertainty: float,
    primary_threshold: float,
    models_out: Dict[str, Any],
    dual_policy_alerts: Dict[str, Any],
) -> tuple[Optional[int], Optional[Dict[str, Any]]]:
    if not persistence.persist_event_type or persistence.persist_in_cooldown:
        return None, None

    margin = float(primary_probability - primary_threshold)
    meta = {
        "dataset": runtime.dataset_code,
        "mode": effective_mode,
        "op_code": runtime.op_code,
        "use_mc": bool(runtime.use_mc),
        "mc_M": int(runtime.mc_M),
        "expected_fps": expected_fps,
        "capture_fps_est": capture_fps_est,
        "models": models_out,
        "policy_alerts": dual_policy_alerts,
        "safe_alert": safe_alert,
        "safe_state": safe_state_out,
        "recall_alert": recall_alert,
        "recall_state": recall_state_out,
        "threshold": primary_threshold,
        "margin": margin,
        "uncertainty": primary_uncertainty,
        "location": event_location,
        "input_source": input_source,
        "event_source": "replay" if input_source in {"video", "replay", "file"} else "realtime",
        "persist_event_type": persistence.persist_event_type,
    }

    saved_event_id = insert_monitor_event(
        conn,
        resident_id=resident_id,
        event_type=str(persistence.persist_event_type),
        severity="high" if persistence.persist_event_type == "fall" else "medium",
        model_code=runtime.active_model_code,
        score=float(p_display),
        meta=meta,
        table_exists=table_exists,
    )

    notification_dispatch = None
    if saved_event_id is not None and persistence.persist_event_type == "fall":
        notification_dispatch = dispatch_fall_notifications(
            conn,
            resident_id=int(resident_id),
            event_id=int(saved_event_id),
            p_fall=float(p_display),
            source="monitor",
        )

    conn.commit()
    session_state[persistence.persist_cd_key] = float(current_t_s)

    try:
        if persistence.persist_event_type == "fall" and runtime.notify_on_every_fall and saved_event_id is not None:
            get_notification_manager().handle_event(
                SafeGuardEvent(
                    event_id=str(saved_event_id),
                    resident_id=int(resident_id),
                    location=event_location,
                    probability=float(primary_probability),
                    uncertainty=float(primary_uncertainty),
                    threshold=float(primary_threshold),
                    margin=float(margin),
                    triage_state=str(triage_state),
                    safe_alert=bool(safe_alert),
                    recall_alert=bool(recall_alert),
                    model_code=str(runtime.active_model_code),
                    dataset_code=str(runtime.dataset_code),
                    op_code=str(runtime.op_code),
                    source="monitor",
                    meta={
                        "event_db_id": int(saved_event_id),
                        "requested_mode": requested_mode,
                        "effective_mode": effective_mode,
                        "safe_state": safe_state_out,
                        "recall_state": recall_state_out,
                    },
                ),
                NotificationPreferences(
                    phone_enabled=bool(runtime.notify_phone),
                    sms_enabled=bool(runtime.notify_sms),
                    email_enabled=True,
                    caregiver_name=runtime.caregiver_name,
                    caregiver_phone=runtime.caregiver_phone,
                    caregiver_email=runtime.caregiver_email,
                ),
            )
    except Exception:
        pass

    return saved_event_id, notification_dispatch
