from __future__ import annotations

"""Final policy stage for monitor prediction responses."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MonitorDecisionResult:
    """Decision payload returned to the route after all alert gates are applied."""

    triage_state: str
    p_display: float
    safe_alert: bool
    recall_alert: bool
    started_event: bool
    low_fps_confirm_count: int
    low_fps_need: int
    low_fps_gate_reason: Optional[str]
    safe_state_out: str
    recall_state_out: str


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _set_policy_state(
    policy_alerts: Dict[str, Any],
    *,
    triage_state: str,
    safe_alert: bool,
    recall_alert: bool,
    started_event: bool,
) -> None:
    for name, alert in (("safe", safe_alert), ("recall", recall_alert)):
        policy = policy_alerts.setdefault(name, {})
        policy["state"] = triage_state
        policy["alert"] = bool(alert)
        policy["started_event"] = bool(started_event and alert and triage_state == "fall")


def _primary_model_context(models_out: Dict[str, Any], p_display: float) -> tuple[Dict[str, Any], Dict[str, Any], float, float]:
    primary_model = models_out["tcn"]
    primary_triage = primary_model["triage"] if isinstance(primary_model.get("triage"), dict) else {}
    tau_high = _finite_float(primary_triage.get("tau_high"))
    if tau_high is None:
        tau_high = _finite_float(primary_model.get("tau_high")) or 0.41
    score_input = _finite_float(primary_triage.get("ps"))
    if score_input is None:
        score_input = _finite_float(primary_model.get("p_alert_in")) or float(p_display)
    return primary_model, primary_triage, float(tau_high), float(score_input)


def _attach_policy_scores(*, primary_model: Dict[str, Any], policy_score: float) -> None:
    primary_model["policy_score"] = float(policy_score)
    raw_model_score = _finite_float(primary_model.get("mu"))
    if raw_model_score is None:
        raw_model_score = _finite_float(primary_model.get("p_det"))
    if raw_model_score is not None:
        primary_model["raw_model_score"] = float(raw_model_score)


def resolve_monitor_decision(
    *,
    models_out: Dict[str, Any],
    tri_tcn: Optional[str],
    policy_alerts: Dict[str, Any],
    dataset_code: str,
    st: Dict[str, Any],
    live_guard: Dict[str, Any],
    low_motion_block: bool,
    recent_motion_support: bool,
    structural_quality_block: bool,
    occlusion_block: bool,
    started_tcn: bool,
    low_fps_mode: bool,
) -> MonitorDecisionResult:
    """Resolve tracker output into one visible monitor decision."""

    safe_policy = policy_alerts["safe"]
    recall_policy = policy_alerts["recall"]
    triage_state = str(safe_policy.get("state") or tri_tcn or "not_fall")
    safe_alert = bool(safe_policy.get("alert"))
    recall_alert = bool(recall_policy.get("alert"))
    started_event = bool(safe_policy.get("started_event") or started_tcn)
    p_display = float(models_out["tcn"].get("p_alert_in", models_out["tcn"].get("mu", 0.0)) or 0.0)

    def apply_decision(state: str, *, safe: bool, recall: bool, started: bool) -> None:
        nonlocal triage_state, safe_alert, recall_alert, started_event
        triage_state = state
        safe_alert = bool(safe)
        recall_alert = bool(recall)
        started_event = bool(started)
        _set_policy_state(
            policy_alerts,
            triage_state=triage_state,
            safe_alert=safe_alert,
            recall_alert=recall_alert,
            started_event=started_event,
        )

    primary_uncertainty_eval = models_out["tcn"].get("uncertainty_gate_eval", {})
    if (
        isinstance(primary_uncertainty_eval, dict)
        and bool(primary_uncertainty_eval.get("blocked_fall", False))
        and triage_state == "fall"
    ):
        apply_decision("uncertain", safe=False, recall=False, started=False)

    if (
        isinstance(primary_uncertainty_eval, dict)
        and bool(primary_uncertainty_eval.get("blocked_fall", False))
        and triage_state == "uncertain"
    ):
        started_event = False

    low_fps_gate_key = f"{dataset_code}:tcn:fall_confirm_count"
    low_fps_confirm_count = int(st.get(low_fps_gate_key, 0) or 0)
    low_fps_gate_reason: Optional[str] = None
    if (
        bool(live_guard["enable_low_motion_gate"])
        and low_motion_block
        and not recent_motion_support
        and triage_state == "fall"
    ):
        apply_decision("uncertain", safe=False, recall=False, started=False)
    if bool(live_guard["enable_occlusion_gate"]) and occlusion_block and triage_state == "fall":
        apply_decision("uncertain", safe=False, recall=False, started=False)
    if bool(live_guard["enable_structural_gate"]) and structural_quality_block and triage_state == "fall":
        apply_decision("uncertain", safe=False, recall=False, started=False)

    low_fps_need = int(live_guard["low_fps_fall_persist_n"])
    if bool(live_guard["enable_low_fps_persist_gate"]) and triage_state == "fall" and low_fps_mode:
        safe_gate_ok = bool(safe_alert)
        motion_gate_ok = not low_motion_block
        structure_gate_ok = not structural_quality_block
        occlusion_gate_ok = not occlusion_block
        if safe_gate_ok and motion_gate_ok and structure_gate_ok and occlusion_gate_ok:
            low_fps_confirm_count += 1
            st[low_fps_gate_key] = low_fps_confirm_count
            if low_fps_confirm_count < low_fps_need:
                low_fps_gate_reason = "need_more_consecutive_fall_windows"
                apply_decision("uncertain", safe=False, recall=False, started=False)
        else:
            low_fps_confirm_count = 0
            st[low_fps_gate_key] = 0
            low_fps_gate_reason = "failed_low_fps_strict_gate"
            apply_decision("uncertain", safe=False, recall=False, started=False)
    else:
        st[low_fps_gate_key] = 0
        low_fps_confirm_count = 0

    primary_model, _, tau_high_live, score_input = _primary_model_context(models_out, p_display)
    p_display = float(score_input)
    _attach_policy_scores(primary_model=primary_model, policy_score=score_input)

    return MonitorDecisionResult(
        triage_state=str(triage_state),
        p_display=float(p_display),
        safe_alert=bool(safe_alert),
        recall_alert=bool(recall_alert),
        started_event=bool(started_event),
        low_fps_confirm_count=int(low_fps_confirm_count),
        low_fps_need=int(low_fps_need),
        low_fps_gate_reason=low_fps_gate_reason,
        safe_state_out=str(policy_alerts["safe"]["state"]),
        recall_state_out=str(policy_alerts["recall"]["state"]),
    )
