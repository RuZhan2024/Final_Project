from __future__ import annotations

"""Model-inference stage for the monitor prediction pipeline.

Inputs to this module are already normalized request fields and a fixed-length
pose window. Outputs must preserve both raw model diagnostics and gated tracker
state because later services decide separately what to display, persist, and
deliver as an alert.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MonitorInferenceResult:
    """Contract between inference and decision stages.

    ``models_out`` keeps the raw model response plus added fields such as
    ``p_alert_in`` and ``triage``. ``started_tcn`` is the tracker edge signal
    used later for persistence.
    """

    models_out: Dict[str, Any]
    tri_tcn: Optional[str]
    policy_alerts: Dict[str, Any]
    low_motion_high_conf_bypass: bool
    started_tcn: bool
    infer_tcn_ms: Optional[int]


def run_monitor_inference(
    *,
    xy,
    conf,
    expected_fps: float,
    target_T: int,
    op_code: str,
    effective_use_mc: bool,
    effective_mc_M: int,
    tcn_key: str,
    dataset_code: str,
    lying_score: Optional[float],
    confirm_motion_score: Optional[float],
    live_guard: Dict[str, Any],
    st: Dict[str, Any],
    st_trackers: Dict[str, Any],
    st_trackers_cfg: Dict[str, Any],
    current_t_s: float,
    low_motion_block: bool,
    recent_motion_support: bool,
    structural_quality_block: bool,
    predict_spec,
    apply_uncertainty_fall_gate,
    tracker_cls,
    low_motion_high_conf_bypass_fn,
) -> MonitorInferenceResult:
    """Run the monitor TCN model and update tracker state.

    Resolution order for each model probability is:
    model output -> live quality gate cap -> uncertainty gate -> temporal
    tracker. The function mutates only the supplied tracker dictionaries; it
    does not persist events or change session state outside those tracker
    objects.
    """

    models_out: Dict[str, Any] = {}
    tri_tcn = None
    policy_alerts: Dict[str, Any] = {}
    low_motion_high_conf_bypass = False
    started_tcn = False
    infer_tcn_ms: Optional[int] = None
    # Keep the original model output intact and feed only the alert input
    # through quality gates, otherwise the response would lose the evidence
    # needed to debug why a confident model prediction was suppressed.
    t_inf = time.perf_counter()
    out_tcn = predict_spec(
        spec_key=tcn_key,
        joints_xy=xy,
        conf=conf,
        fps=float(expected_fps),
        target_T=target_T,
        op_code=op_code,
        use_mc=effective_use_mc,
        mc_M=effective_mc_M,
    )

    cfg_tcn = out_tcn.get("alert_cfg") or {}
    tau_low_tcn = float(cfg_tcn.get("tau_low", out_tcn.get("tau_low", 0.0)))
    tau_high_tcn = float(cfg_tcn.get("tau_high", out_tcn.get("tau_high", 0.0)))
    p_raw_tcn = float(out_tcn.get("mu") if out_tcn.get("mu") is not None else out_tcn.get("p_det", 0.0))
    sigma_tcn = float(out_tcn.get("sigma", 0.0) or 0.0)
    uncertainty_gate_tcn = out_tcn.get("uncertainty_gate") if isinstance(out_tcn.get("uncertainty_gate"), dict) else {}
    # A bypass requires both a policy opt-in and repeated high-confidence
    # evidence; this prevents one static high score from defeating the
    # low-motion false-positive guard.
    low_motion_high_conf_bypass = low_motion_high_conf_bypass_fn(
        st,
        dataset_code=dataset_code,
        mode="tcn",
        p_raw=p_raw_tcn,
        tau_high=tau_high_tcn,
        lying_score=lying_score,
        enabled=bool(live_guard.get("allow_low_motion_high_conf_bypass", False)),
        min_hits=int(live_guard.get("low_motion_high_conf_k", 0)),
        max_lying=live_guard.get("low_motion_high_conf_max_lying"),
    )
    if (
        (
            bool(live_guard["enable_low_motion_gate"])
            and low_motion_block
            and not recent_motion_support
            and not low_motion_high_conf_bypass
        )
        or (bool(live_guard["enable_structural_gate"]) and structural_quality_block)
    ):
        # Cap rather than zero the probability so tracker diagnostics still
        # show whether the suppressed window was borderline or severe.
        p_alert_tcn = float(min(float(p_raw_tcn), float(tau_low_tcn) - 0.02))
    else:
        p_alert_tcn = float(p_raw_tcn)
    p_alert_tcn, uncertainty_eval_tcn = apply_uncertainty_fall_gate(
        probability=float(p_alert_tcn),
        sigma=float(sigma_tcn),
        tau_low=float(tau_low_tcn),
        tau_high=float(tau_high_tcn),
        mc_applied=bool(out_tcn.get("mc_applied", False)),
        uncertainty_cfg=uncertainty_gate_tcn,
    )
    out_tcn["uncertainty_gate_eval"] = uncertainty_eval_tcn
    out_tcn["p_alert_in"] = float(p_alert_tcn)
    out_tcn["lying_score"] = None if lying_score is None else float(lying_score)
    out_tcn["confirm_motion_score"] = None if confirm_motion_score is None else float(confirm_motion_score)
    trk = st_trackers.get(tcn_key)
    if trk is None or st_trackers_cfg.get(tcn_key) != cfg_tcn:
        # Tracker state is valid only for the alert config that created it;
        # reusing EMA/cooldown state after an op switch would bias the next
        # decision window.
        trk = tracker_cls(cfg_tcn)
        st_trackers[tcn_key] = trk
        st_trackers_cfg[tcn_key] = cfg_tcn
    r = trk.step(
        p=float(p_alert_tcn),
        t_s=current_t_s,
    )
    out_tcn["triage"] = {
        "state": r.triage_state,
        "ps": r.ps,
        "p_in": r.p_in,
        "tau_low": tau_low_tcn,
        "tau_high": tau_high_tcn,
        "ema_alpha": float(cfg_tcn.get("ema_alpha", 0.0)),
        "k": int(cfg_tcn.get("k", 2)),
        "n": int(cfg_tcn.get("n", 3)),
        "cooldown_s": float(cfg_tcn.get("cooldown_s", 0.0)),
        "cooldown_remaining_s": r.cooldown_remaining_s,
    }
    models_out["tcn"] = out_tcn
    tri_tcn = r.triage_state
    started_tcn = bool(r.started_event)
    infer_tcn_ms = int((time.perf_counter() - t_inf) * 1000)
    policy_alerts = {
        "safe": {
            "state": r.triage_state,
            "alert": bool(r.triage_state == "fall"),
            "started_event": bool(r.started_event),
            "tau_low": float(tau_low_tcn),
            "tau_high": float(tau_high_tcn),
            "cooldown_remaining_s": r.cooldown_remaining_s,
        },
        "recall": {
            "state": r.triage_state,
            "alert": bool(r.triage_state == "fall"),
            "started_event": bool(r.started_event),
            "tau_low": float(tau_low_tcn),
            "tau_high": float(tau_high_tcn),
            "cooldown_remaining_s": r.cooldown_remaining_s,
        },
    }

    return MonitorInferenceResult(
        models_out=models_out,
        tri_tcn=tri_tcn,
        policy_alerts=policy_alerts,
        low_motion_high_conf_bypass=low_motion_high_conf_bypass,
        started_tcn=started_tcn,
        infer_tcn_ms=infer_tcn_ms,
    )
