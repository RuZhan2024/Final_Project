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
    ``p_alert_in`` and ``triage``. ``started_*`` flags are edge signals from the
    model-specific trackers; hybrid mode may still recompute its own edge from
    the combined policy in the decision stage.
    """

    models_out: Dict[str, Any]
    tri_tcn: Optional[str]
    tri_gcn: Optional[str]
    dual_policy_alerts: Dict[str, Any]
    low_motion_high_conf_bypass: bool
    started_tcn: bool
    started_gcn: bool
    infer_tcn_ms: Optional[int]
    infer_gcn_ms: Optional[int]


def run_monitor_inference(
    *,
    mode: str,
    xy,
    conf,
    expected_fps: float,
    target_T: int,
    op_code: str,
    effective_use_mc: bool,
    effective_mc_M: int,
    tcn_key: str,
    gcn_key: str,
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
    load_dual_policy_cfg,
    apply_uncertainty_fall_gate,
    tracker_cls,
    low_motion_high_conf_bypass_fn,
) -> MonitorInferenceResult:
    """Run the requested model family/families and update tracker state.

    Resolution order for each model probability is:
    model output -> live quality gate cap -> uncertainty gate -> temporal
    tracker. The function mutates only the supplied tracker dictionaries; it
    does not persist events, choose final hybrid policy, or change session state
    outside those tracker objects.
    """

    models_out: Dict[str, Any] = {}
    tri_tcn = None
    tri_gcn = None
    dual_policy_alerts: Dict[str, Any] = {}
    low_motion_high_conf_bypass = False
    started_tcn = False
    started_gcn = False
    infer_tcn_ms: Optional[int] = None
    infer_gcn_ms: Optional[int] = None

    run_tcn = mode in {"tcn", "hybrid"}
    run_gcn = mode in {"gcn", "hybrid"}

    if run_tcn:
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
        bypass_tcn = low_motion_high_conf_bypass_fn(
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
        low_motion_high_conf_bypass = bool(low_motion_high_conf_bypass or bypass_tcn)
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

        for pol in ("safe", "recall"):
            # The comparison is policy-only: both trackers see the same model
            # probability so differences come from thresholds/cooldowns, not a
            # second inference pass.
            pol_cfg = load_dual_policy_cfg(dataset_code, pol, op_code)
            if not isinstance(pol_cfg, dict):
                continue
            pol_key = f"{tcn_key}::dual::{pol}"
            pol_trk = st_trackers.get(pol_key)
            if pol_trk is None or st_trackers_cfg.get(pol_key) != pol_cfg:
                pol_trk = tracker_cls(pol_cfg)
                st_trackers[pol_key] = pol_trk
                st_trackers_cfg[pol_key] = pol_cfg

            pol_res = pol_trk.step(
                p=float(p_alert_tcn),
                t_s=current_t_s,
            )
            dual_policy_alerts[pol] = {
                "state": pol_res.triage_state,
                "alert": bool(pol_res.triage_state == "fall"),
                "started_event": bool(pol_res.started_event),
                "tau_low": float(pol_cfg.get("tau_low", 0.0)),
                "tau_high": float(pol_cfg.get("tau_high", 0.0)),
                "cooldown_remaining_s": pol_res.cooldown_remaining_s,
            }

    if run_gcn:
        # GCN cannot share the TCN tracker even in hybrid mode; its thresholds,
        # smoothing, and cooldown can be calibrated from a different spec.
        t_inf = time.perf_counter()
        out_gcn = predict_spec(
            spec_key=gcn_key,
            joints_xy=xy,
            conf=conf,
            fps=float(expected_fps),
            target_T=target_T,
            op_code=op_code,
            use_mc=effective_use_mc,
            mc_M=effective_mc_M,
        )
        models_out["gcn"] = out_gcn

        cfg_gcn = out_gcn.get("alert_cfg") or {}
        tau_low_gcn = float(cfg_gcn.get("tau_low", out_gcn.get("tau_low", 0.0)))
        tau_high_gcn = float(cfg_gcn.get("tau_high", out_gcn.get("tau_high", 0.0)))
        p_raw_gcn = float(out_gcn.get("mu") or out_gcn.get("p_det") or 0.0)
        sigma_gcn = float(out_gcn.get("sigma", 0.0) or 0.0)
        uncertainty_gate_gcn = out_gcn.get("uncertainty_gate") if isinstance(out_gcn.get("uncertainty_gate"), dict) else {}
        bypass_gcn = low_motion_high_conf_bypass_fn(
            st,
            dataset_code=dataset_code,
            mode="gcn",
            p_raw=p_raw_gcn,
            tau_high=tau_high_gcn,
            lying_score=lying_score,
            enabled=bool(live_guard.get("allow_low_motion_high_conf_bypass", False)),
            min_hits=int(live_guard.get("low_motion_high_conf_k", 0)),
            max_lying=live_guard.get("low_motion_high_conf_max_lying"),
        )
        low_motion_high_conf_bypass = bool(low_motion_high_conf_bypass or bypass_gcn)
        if (
            (
                bool(live_guard["enable_low_motion_gate"])
                and low_motion_block
                and not recent_motion_support
                and not low_motion_high_conf_bypass
            )
            or (bool(live_guard["enable_structural_gate"]) and structural_quality_block)
        ):
            p_alert_gcn = float(min(float(p_raw_gcn), float(tau_low_gcn) - 0.02))
        else:
            p_alert_gcn = float(p_raw_gcn)
        p_alert_gcn, uncertainty_eval_gcn = apply_uncertainty_fall_gate(
            probability=float(p_alert_gcn),
            sigma=float(sigma_gcn),
            tau_low=float(tau_low_gcn),
            tau_high=float(tau_high_gcn),
            mc_applied=bool(out_gcn.get("mc_applied", False)),
            uncertainty_cfg=uncertainty_gate_gcn,
        )
        out_gcn["uncertainty_gate_eval"] = uncertainty_eval_gcn
        out_gcn["p_alert_in"] = float(p_alert_gcn)
        trk = st_trackers.get(gcn_key)
        if trk is None or st_trackers_cfg.get(gcn_key) != cfg_gcn:
            # Operating-point changes invalidate the old tracker state here too.
            trk = tracker_cls(cfg_gcn)
            st_trackers[gcn_key] = trk
            st_trackers_cfg[gcn_key] = cfg_gcn
        res = trk.step(p=float(p_alert_gcn), t_s=current_t_s)
        out_gcn["triage"] = {
            "state": res.triage_state,
            "ps": res.ps,
            "p_in": res.p_in,
            "tau_low": tau_low_gcn,
            "tau_high": tau_high_gcn,
            "ema_alpha": float(cfg_gcn.get("ema_alpha", 0.0)),
            "k": int(cfg_gcn.get("k", 2)),
            "n": int(cfg_gcn.get("n", 3)),
            "cooldown_s": float(cfg_gcn.get("cooldown_s", 0.0)),
            "cooldown_remaining_s": res.cooldown_remaining_s,
        }
        started_gcn = bool(res.started_event)
        tri_gcn = res.triage_state
        infer_gcn_ms = int((time.perf_counter() - t_inf) * 1000)

    return MonitorInferenceResult(
        models_out=models_out,
        tri_tcn=tri_tcn,
        tri_gcn=tri_gcn,
        dual_policy_alerts=dual_policy_alerts,
        low_motion_high_conf_bypass=low_motion_high_conf_bypass,
        started_tcn=started_tcn,
        started_gcn=started_gcn,
        infer_tcn_ms=infer_tcn_ms,
        infer_gcn_ms=infer_gcn_ms,
    )
