from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..code_normalization import norm_op_code


@dataclass(frozen=True)
class MonitorDecisionResult:
    triage_state: str
    p_display: float
    safe_alert: bool
    recall_alert: bool
    started_event: bool
    low_fps_confirm_count: int
    low_fps_need: int
    low_fps_gate_reason: Optional[str]
    delivery_gate_diag: Dict[str, Any]
    uncertain_promoted: bool
    safe_state_out: str
    recall_state_out: str


def resolve_monitor_decision(
    *,
    mode: str,
    models_out: Dict[str, Any],
    tri_tcn: Optional[str],
    tri_gcn: Optional[str],
    dual_policy_alerts: Dict[str, Any],
    primary_model_key: str,
    primary_spec_key: str,
    resident_id: int,
    dataset_code: str,
    op_code: str,
    st: Dict[str, Any],
    current_t_s: float,
    is_replay: bool,
    live_guard: Dict[str, Any],
    delivery_gate: Dict[str, Any],
    uncertain_promote: Dict[str, Any],
    low_motion_block: bool,
    recent_motion_support: bool,
    low_motion_high_conf_bypass: bool,
    structural_quality_block: bool,
    occlusion_block: bool,
    lying_score: Optional[float],
    confirm_motion_score: Optional[float],
    started_tcn: bool,
    started_gcn: bool,
    low_fps_mode: bool,
) -> MonitorDecisionResult:
    tcn_safe_alert = (
        bool(dual_policy_alerts.get("safe", {}).get("alert"))
        if "safe" in dual_policy_alerts
        else bool((tri_tcn or "not_fall") == "fall")
    )
    tcn_recall_alert = (
        bool(dual_policy_alerts.get("recall", {}).get("alert"))
        if "recall" in dual_policy_alerts
        else tcn_safe_alert
    )
    gcn_alert = bool((tri_gcn or "not_fall") == "fall")

    if mode == "tcn":
        triage_state = str(dual_policy_alerts.get("safe", {}).get("state") or tri_tcn or "not_fall")
        p_display = float(models_out.get("tcn", {}).get("p_alert_in", models_out.get("tcn", {}).get("mu", 0.0)))
        safe_alert = tcn_safe_alert
        recall_alert = tcn_recall_alert
    elif mode == "gcn":
        triage_state = tri_gcn or "not_fall"
        p_display = float(models_out.get("gcn", {}).get("p_alert_in", models_out.get("gcn", {}).get("mu", 0.0)))
        safe_alert = gcn_alert
        recall_alert = gcn_alert
    else:
        safe_alert = bool(tcn_safe_alert and gcn_alert)
        recall_alert = bool(tcn_recall_alert or gcn_alert)
        if safe_alert:
            triage_state = "fall"
        elif recall_alert:
            triage_state = "uncertain"
        else:
            triage_state = "not_fall"
        p_tcn = float(models_out.get("tcn", {}).get("p_alert_in", models_out.get("tcn", {}).get("mu", 0.0)))
        p_gcn = float(models_out.get("gcn", {}).get("p_alert_in", models_out.get("gcn", {}).get("mu", 0.0)))
        p_display = float(max(p_tcn, p_gcn))
        dual_policy_alerts.setdefault(
            "safe",
            {
                "state": "fall" if safe_alert else "not_fall",
                "alert": safe_alert,
                "source": "tcn_safe_and_gcn",
            },
        )
        dual_policy_alerts.setdefault(
            "recall",
            {
                "state": "fall" if recall_alert else "not_fall",
                "alert": recall_alert,
                "source": "tcn_recall_or_gcn",
            },
        )

    primary_uncertainty_eval = (
        models_out.get(primary_model_key, {}).get("uncertainty_gate_eval", {})
        if isinstance(models_out.get(primary_model_key), dict)
        else {}
    )
    if (
        isinstance(primary_uncertainty_eval, dict)
        and bool(primary_uncertainty_eval.get("blocked_fall", False))
        and triage_state == "fall"
    ):
        triage_state = "uncertain"
        safe_alert = False
        recall_alert = False

    if mode == "tcn":
        if "safe" in dual_policy_alerts:
            started_event = bool(dual_policy_alerts.get("safe", {}).get("started_event"))
        else:
            started_event = bool(started_tcn)
    elif mode == "gcn":
        started_event = bool(started_gcn)
    else:
        edge_key = f"persist_edge:{resident_id}:{dataset_code}:{mode}:{op_code}"
        prev_safe = bool(st.get(edge_key, False))
        curr_safe = bool(safe_alert)
        started_event = bool(curr_safe and (not prev_safe))
        st[edge_key] = curr_safe

    if (
        isinstance(primary_uncertainty_eval, dict)
        and bool(primary_uncertainty_eval.get("blocked_fall", False))
        and triage_state == "uncertain"
    ):
        started_event = False

    low_fps_gate_key = f"{dataset_code}:{mode}:fall_confirm_count"
    low_fps_confirm_count = int(st.get(low_fps_gate_key, 0) or 0)
    low_fps_gate_reason: Optional[str] = None
    if (
        bool(live_guard["enable_low_motion_gate"])
        and low_motion_block
        and (not recent_motion_support)
        and (not low_motion_high_conf_bypass)
        and triage_state == "fall"
    ):
        triage_state = "uncertain"
        started_event = False
        safe_alert = False
        recall_alert = False
    if bool(live_guard["enable_occlusion_gate"]) and occlusion_block and triage_state == "fall":
        triage_state = "uncertain"
        started_event = False
        safe_alert = False
        recall_alert = False
    if bool(live_guard["enable_structural_gate"]) and structural_quality_block and triage_state == "fall":
        triage_state = "uncertain"
        started_event = False
        safe_alert = False
        recall_alert = False

    low_fps_need = int(live_guard["low_fps_fall_persist_n"])
    if bool(live_guard["enable_low_fps_persist_gate"]) and triage_state == "fall" and low_fps_mode:
        safe_gate_ok = bool(safe_alert) if mode in {"tcn", "hybrid"} else True
        motion_gate_ok = not low_motion_block
        structure_gate_ok = not structural_quality_block
        occlusion_gate_ok = not occlusion_block
        if safe_gate_ok and motion_gate_ok and structure_gate_ok and occlusion_gate_ok:
            low_fps_confirm_count += 1
            st[low_fps_gate_key] = low_fps_confirm_count
            if low_fps_confirm_count < low_fps_need:
                low_fps_gate_reason = "need_more_consecutive_fall_windows"
                triage_state = "uncertain"
                started_event = False
                safe_alert = False
                recall_alert = False
        else:
            low_fps_confirm_count = 0
            st[low_fps_gate_key] = 0
            low_fps_gate_reason = "failed_low_fps_strict_gate"
            triage_state = "uncertain"
            started_event = False
            safe_alert = False
            recall_alert = False
    else:
        st[low_fps_gate_key] = 0
        low_fps_confirm_count = 0

    delivery_gate_diag: Dict[str, Any] = {
        "enabled": bool(delivery_gate.get("enabled", False)),
        "blocked": False,
        "reason": None,
        "start_lying": None,
        "max_lying": None,
        "mean_motion_high": None,
        "event_start_s": None,
    }
    if mode in {"tcn", "gcn"} and bool(delivery_gate.get("enabled", False)):
        gate_state_key = f"delivery_gate:{primary_spec_key}:{norm_op_code(op_code)}"
        if triage_state == "fall":
            gate_state = st.setdefault(gate_state_key, {})
            if started_event or not bool(gate_state.get("active", False)):
                gate_state.clear()
                gate_state["active"] = True
                gate_state["event_start_t_s"] = float(current_t_s)
                gate_state["start_lying"] = (
                    float(lying_score) if lying_score is not None and math.isfinite(float(lying_score)) else float("-inf")
                )
                gate_state["max_lying"] = (
                    float(lying_score) if lying_score is not None and math.isfinite(float(lying_score)) else float("-inf")
                )
                gate_state["motion_high_sum"] = 0.0
                gate_state["motion_high_count"] = 0
            if lying_score is not None and math.isfinite(float(lying_score)):
                gate_state["max_lying"] = max(float(gate_state.get("max_lying", float("-inf"))), float(lying_score))
            tau_high_live = float(models_out.get(primary_model_key, {}).get("triage", {}).get("tau_high", 1.0))
            if (
                confirm_motion_score is not None
                and math.isfinite(float(confirm_motion_score))
                and float(models_out.get(primary_model_key, {}).get("p_alert_in", 0.0)) >= tau_high_live
            ):
                gate_state["motion_high_sum"] = float(gate_state.get("motion_high_sum", 0.0)) + float(confirm_motion_score)
                gate_state["motion_high_count"] = int(gate_state.get("motion_high_count", 0)) + 1

            mean_motion_high = None
            if int(gate_state.get("motion_high_count", 0)) > 0:
                mean_motion_high = float(gate_state["motion_high_sum"]) / float(gate_state["motion_high_count"])
            event_start_s = float(gate_state.get("event_start_t_s", current_t_s)) - float(st.get("session_start_t_s", current_t_s))
            max_lying_seen = gate_state.get("max_lying")
            start_lying_seen = gate_state.get("start_lying")
            if isinstance(start_lying_seen, (int, float)) and math.isfinite(float(start_lying_seen)):
                delivery_gate_diag["start_lying"] = float(start_lying_seen)
            if isinstance(max_lying_seen, (int, float)) and math.isfinite(float(max_lying_seen)):
                delivery_gate_diag["max_lying"] = float(max_lying_seen)
            delivery_gate_diag["mean_motion_high"] = mean_motion_high
            delivery_gate_diag["event_start_s"] = float(event_start_s)

            gate_reason = None
            if (
                delivery_gate.get("max_start_lying") is not None
                and delivery_gate_diag["start_lying"] is not None
                and delivery_gate_diag["start_lying"] > float(delivery_gate["max_start_lying"])
            ):
                gate_reason = "start_lying"
            elif (
                delivery_gate.get("max_lying") is not None
                and delivery_gate_diag["max_lying"] is not None
                and delivery_gate_diag["max_lying"] > float(delivery_gate["max_lying"])
            ):
                gate_reason = "max_lying"
            elif (
                delivery_gate.get("min_mean_motion_high") is not None
                and mean_motion_high is not None
                and mean_motion_high < float(delivery_gate["min_mean_motion_high"])
            ):
                gate_reason = "mean_motion_high"
            elif (
                delivery_gate.get("max_event_start_s") is not None
                and event_start_s > float(delivery_gate["max_event_start_s"])
            ):
                gate_reason = "event_start_s"

            if gate_reason is not None:
                triage_state = "uncertain"
                started_event = False
                safe_alert = False
                recall_alert = False
                delivery_gate_diag["blocked"] = True
                delivery_gate_diag["reason"] = gate_reason
        else:
            st.pop(gate_state_key, None)

    uncertain_promoted = False
    if (
        mode in {"tcn", "gcn"}
        and triage_state == "uncertain"
        and bool(uncertain_promote.get("enabled", False))
        and (not bool(uncertain_promote.get("video_only", True)) or is_replay)
    ):
        p_alert_live = float(models_out.get(primary_model_key, {}).get("p_alert_in", 0.0) or 0.0)
        p_ok = (
            uncertain_promote.get("min_p_alert") is None
            or p_alert_live >= float(uncertain_promote["min_p_alert"])
        )
        motion_ok = (
            uncertain_promote.get("min_motion") is None
            or (
                confirm_motion_score is not None
                and math.isfinite(float(confirm_motion_score))
                and float(confirm_motion_score) >= float(uncertain_promote["min_motion"])
            )
        )
        lying_ok = (
            uncertain_promote.get("max_lying") is None
            or (
                lying_score is not None
                and math.isfinite(float(lying_score))
                and float(lying_score) <= float(uncertain_promote["max_lying"])
            )
        )
        if p_ok and motion_ok and lying_ok:
            triage_state = "fall"
            safe_alert = True
            recall_alert = True
            uncertain_promoted = True

    safe_state_out = dual_policy_alerts.get("safe", {}).get("state")
    recall_state_out = dual_policy_alerts.get("recall", {}).get("state")
    if not safe_state_out:
        if triage_state == "uncertain":
            safe_state_out = "uncertain"
        else:
            safe_state_out = "fall" if bool(safe_alert) else "not_fall"
    if not recall_state_out:
        if triage_state == "uncertain":
            recall_state_out = "uncertain"
        else:
            recall_state_out = "fall" if bool(recall_alert) else "not_fall"

    return MonitorDecisionResult(
        triage_state=str(triage_state),
        p_display=float(p_display),
        safe_alert=bool(safe_alert),
        recall_alert=bool(recall_alert),
        started_event=bool(started_event),
        low_fps_confirm_count=int(low_fps_confirm_count),
        low_fps_need=int(low_fps_need),
        low_fps_gate_reason=low_fps_gate_reason,
        delivery_gate_diag=delivery_gate_diag,
        uncertain_promoted=bool(uncertain_promoted),
        safe_state_out=str(safe_state_out),
        recall_state_out=str(recall_state_out),
    )
