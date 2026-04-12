from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone

from typing import Any, Dict, List, Optional
import numpy as np

from fastapi import APIRouter, Body, HTTPException, Query, WebSocket, WebSocketDisconnect

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from ..code_normalization import norm_op_code, normalize_dataset_code
from ..db import get_conn
from ..db_schema import ensure_system_settings_schema, table_exists
from ..deploy_ops import detect_variants
from ..deploy_runtime import (
    get_pose_preprocess_cfg as _get_pose_preprocess_cfg,
    get_specs as _get_deploy_specs,
    predict_spec as _predict_spec,
)
from ..monitor_policy import (
    load_dual_policy_cfg as _load_dual_policy_cfg,
    op_delivery_gate as _op_delivery_gate,
    op_live_guard as _op_live_guard,
    op_uncertain_promote as _op_uncertain_promote,
    resolve_monitor_specs as _resolve_monitor_specs,
)
from ..monitor_windowing import (
    decode_quantized_raw_window as _decode_quantized_raw_window,
    direct_window_stats as _direct_window_stats,
    raw_window_stats as _raw_window_stats,
    preprocess_online_raw_window as _preprocess_online_raw_window,
    resolve_runtime_fps as _resolve_runtime_fps,
    resample_pose_window as _resample_pose_window,
    window_motion_score as _window_motion_score,
    window_quality_block as _window_quality_block,
)
from ..online_alert import OnlineAlertTracker
from ..schemas import MonitorPredictPayload
from ..services.monitor_response_service import (
    build_monitor_prediction_response,
    build_stale_monitor_response,
    log_monitor_perf_if_slow,
)
from ..services.monitor_request_service import prepare_monitor_request
from ..services.monitor_uncertainty_service import apply_uncertainty_fall_gate
from ..services.monitor_runtime_service import (
    persist_monitor_event,
    resolve_monitor_persistence_plan,
)
from ..services.monitor_session_service import reset_monitor_session
from ..services.value_coercion import coerce_bool
from ..runtime_state import get_session_store, set_last_prediction_snapshot
from fall_detection.deploy.common import WindowRaw, compute_confirm_scores


router = APIRouter()
logger = logging.getLogger(__name__)
_LOW_MOTION_MEMORY_WINDOWS = 5


_detect_variants = detect_variants
_ensure_system_settings_schema = ensure_system_settings_schema
_table_exists = table_exists
_norm_op_code = norm_op_code


def _recent_motion_support(
    st: Dict[str, Any],
    *,
    dataset_code: str,
    mode: str,
    motion_score: Optional[float],
    min_motion: float,
    memory_windows: int = _LOW_MOTION_MEMORY_WINDOWS,
) -> bool:
    """Remember short-term motion bursts so post-fall stillness is not misread.

    A real fall often has one or two high-motion windows followed by low motion
    while the person is lying on the floor. We keep a tiny per-session history
    and allow that recent burst to support the current window.
    """
    hist_key = f"motion_hist:{dataset_code}:{mode}"
    hist_raw = st.get(hist_key)
    hist: List[float] = hist_raw if isinstance(hist_raw, list) else []
    if motion_score is not None and math.isfinite(float(motion_score)):
        hist.append(float(motion_score))
    keep = max(1, int(memory_windows))
    if len(hist) > keep:
        hist = hist[-keep:]
    st[hist_key] = hist
    return any(float(v) >= float(min_motion) for v in hist)


def _low_motion_high_conf_bypass(
    st: Dict[str, Any],
    *,
    dataset_code: str,
    mode: str,
    p_raw: float,
    tau_high: float,
    lying_score: Optional[float],
    enabled: bool,
    min_hits: int,
    max_lying: Optional[float],
) -> bool:
    """Allow specific low-motion clips through when confidence stays high.

    This is meant for scenes where true falls can become nearly static very
    quickly. We require a short streak of high-confidence windows, and if a
    max-lying guard is configured every window in the streak must stay below it.
    """
    key = f"low_motion_high_conf:{dataset_code}:{mode}"
    if not enabled or min_hits <= 0:
        st.pop(key, None)
        return False

    prev = st.get(key)
    count = int(prev.get("count", 0) or 0) if isinstance(prev, dict) else 0
    current_ok = bool(float(p_raw) >= float(tau_high))
    if current_ok and max_lying is not None:
        if lying_score is None or (not math.isfinite(float(lying_score))) or float(lying_score) > float(max_lying):
            current_ok = False

    if current_ok:
        count += 1
        st[key] = {"count": count}
    else:
        st.pop(key, None)
        count = 0
    return bool(count >= int(min_hits))

def _compact_model_out(model_out: Any) -> Dict[str, Any]:
    src = model_out if isinstance(model_out, dict) else {}
    tri = src.get("triage") if isinstance(src.get("triage"), dict) else {}
    out: Dict[str, Any] = {}
    for key in ("mu", "sigma", "p_alert_in", "p_det"):
        if src.get(key) is not None:
            out[key] = src.get(key)
    tri_out: Dict[str, Any] = {}
    for key in ("state", "ps", "tau_low", "tau_high"):
        if tri.get(key) is not None:
            tri_out[key] = tri.get(key)
    if tri_out:
        out["triage"] = tri_out
    return out


def _compact_policy_alerts(policy_alerts: Any) -> Dict[str, Any]:
    src = policy_alerts if isinstance(policy_alerts, dict) else {}
    out: Dict[str, Any] = {}
    for name in ("safe", "recall"):
        pol = src.get(name)
        if not isinstance(pol, dict):
            continue
        out[name] = {
            "state": pol.get("state"),
            "alert": pol.get("alert"),
        }
    return out


def _compact_monitor_response(resp: Dict[str, Any], mode: str) -> Dict[str, Any]:
    models = resp.get("models") if isinstance(resp.get("models"), dict) else {}
    mode_l = str(mode or "").lower()
    compact_models: Dict[str, Any] = {}
    if mode_l == "hybrid":
        if "tcn" in models:
            compact_models["tcn"] = _compact_model_out(models.get("tcn"))
        if "gcn" in models:
            compact_models["gcn"] = _compact_model_out(models.get("gcn"))
    elif mode_l in {"tcn", "gcn"} and mode_l in models:
        compact_models[mode_l] = _compact_model_out(models.get(mode_l))

    return {
        "triage_state": resp.get("triage_state"),
        "safe_alert": resp.get("safe_alert"),
        "safe_state": resp.get("safe_state"),
        "recall_alert": resp.get("recall_alert"),
        "recall_state": resp.get("recall_state"),
        "event_id": resp.get("event_id"),
        "window_end_t_ms": resp.get("window_end_t_ms"),
        "stale_drop": resp.get("stale_drop"),
        "stale_reason": resp.get("stale_reason"),
        "models": compact_models,
        "policy_alerts": _compact_policy_alerts(resp.get("policy_alerts")),
    }




@router.post("/api/monitor/reset_session")
@router.post("/api/v1/monitor/reset_session")
def reset_session(session_id: str = Query(...)) -> Dict[str, Any]:
    return reset_monitor_session(get_session_store(), session_id)


@router.post("/api/monitor/predict_window")
@router.post("/api/v1/monitor/predict_window")
def predict_window(payload: MonitorPredictPayload = Body(...)) -> Dict[str, Any]:
    """Score one window from the live monitor UI."""
    t0 = time.time()
    perf_started = time.perf_counter()
    perf_last = perf_started
    perf: Dict[str, int] = {}

    def _mark_perf(name: str) -> None:
        nonlocal perf_last
        now = time.perf_counter()
        perf[name] = int((now - perf_last) * 1000)
        perf_last = now

    prepared = prepare_monitor_request(
        payload=payload,
        logger=logger,
        get_conn=get_conn,
        normalize_dataset_code=normalize_dataset_code,
        coerce_bool=coerce_bool,
        decode_quantized_raw_window=_decode_quantized_raw_window,
        raw_window_stats=_raw_window_stats,
        resolve_runtime_fps=_resolve_runtime_fps,
        resolve_monitor_specs=_resolve_monitor_specs,
        get_pose_preprocess_cfg=_get_pose_preprocess_cfg,
        resample_pose_window=_resample_pose_window,
        preprocess_online_raw_window=_preprocess_online_raw_window,
        direct_window_stats=_direct_window_stats,
        get_deploy_specs=_get_deploy_specs,
        ensure_system_settings_schema=_ensure_system_settings_schema,
        detect_variants=_detect_variants,
        table_exists=_table_exists,
    )
    compact_response = prepared.compact_response
    _mark_perf("parse_inputs_ms")
    _mark_perf("db_defaults_ms")

    session_id = prepared.session_id
    input_source = prepared.input_source
    is_replay = prepared.is_replay
    requested_mode = prepared.requested_mode
    mode = prepared.mode
    dataset_code = prepared.dataset_code
    op_code = prepared.op_code
    requested_use_mc = prepared.requested_use_mc
    requested_mc_M = prepared.requested_mc_M
    effective_use_mc = prepared.effective_use_mc
    effective_mc_M = prepared.effective_mc_M
    persist = prepared.persist
    target_T = prepared.target_T
    raw_stats = prepared.raw_stats
    xy = prepared.xy
    conf = prepared.conf
    cap_fps_est = prepared.cap_fps_est
    resident_id = prepared.resident_id
    active_model_code = prepared.active_model_code
    cooldown_sec = prepared.cooldown_sec
    runtime = prepared.runtime
    event_location = prepared.event_location
    expected_fps = prepared.expected_fps
    effective_fps = prepared.effective_fps
    st = prepared.session_state
    _t_s = prepared.current_t_s
    specs = prepared.specs
    tcn_key = prepared.tcn_key
    gcn_key = prepared.gcn_key
    guard_spec_key = prepared.guard_spec_key
    primary_spec_key = prepared.primary_spec_key
    primary_model_key = prepared.primary_model_key
    window_end_t_ms = prepared.window_end_t_ms
    window_seq = prepared.window_seq
    _mark_perf("window_prepare_ms")

    motion_score = _window_motion_score(xy)
    lying_score = None
    confirm_motion_score = None

    try:
        if window_end_t_ms is not None:
            _now_ms = float(window_end_t_ms)
        elif raw_t_ms is not None and len(raw_t_ms) > 0:
            _now_ms = float(raw_t_ms[-1])
        else:
            _now_ms = time.time() * 1000.0
    except (TypeError, ValueError):
        _now_ms = time.time() * 1000.0
    _t_s = float(_now_ms) / 1000.0

    try:
        xy_np = np.asarray(xy, dtype=np.float32)
        conf_np = np.asarray(conf, dtype=np.float32) if conf else None
        raw_window = WindowRaw(
            joints_xy=xy_np,
            motion_xy=None,
            conf=conf_np,
            mask=None,
            fps=float(effective_fps),
            meta=None,
        )
        lying_score, confirm_motion_score = compute_confirm_scores(raw_window)
    except Exception:
        lying_score, confirm_motion_score = None, None
    _mark_perf("confirm_scores_ms")

    st_trackers = st["trackers"]
    st_trackers_cfg = st["trackers_cfg"]
    started_tcn = False
    started_gcn = False

    live_guard = _op_live_guard(specs, guard_spec_key, op_code, dataset_code)
    delivery_gate = _op_delivery_gate(specs, primary_spec_key, op_code)
    uncertain_promote = _op_uncertain_promote(specs, primary_spec_key, op_code)
    min_motion = float(live_guard["min_motion_for_fall"])
    low_motion_block = bool(motion_score is not None and motion_score < min_motion)
    recent_motion_support = _recent_motion_support(
        st,
        dataset_code=dataset_code,
        mode=mode,
        motion_score=motion_score,
        min_motion=min_motion,
    )
    qdiag = _window_quality_block(
        raw_stats=raw_stats,
        expected_fps=float(expected_fps),
        effective_fps=float(effective_fps),
        target_T=int(target_T),
        dataset_code=dataset_code,
        min_fps_ratio_override=float(live_guard["min_fps_ratio"]),
        min_frames_ratio_override=float(live_guard["min_frames_ratio"]),
        min_coverage_ratio_override=float(live_guard["min_coverage_ratio"]),
    )
    structural_quality_block = bool(qdiag.get("low_frames", False) or qdiag.get("low_coverage", False))
    low_quality_block = bool(qdiag.get("low_quality_block", False))
    raw_fps_est = raw_stats.get("raw_fps_est")
    low_fps_mode = bool(
        raw_fps_est is not None
        and math.isfinite(float(raw_fps_est))
        and float(raw_fps_est) < float(live_guard["low_fps_mode_threshold"])
    )
    sampling_mode = "low_fps" if low_fps_mode else "normal"
    conf_mean = raw_stats.get("conf_mean")
    joints_med = int(raw_stats.get("joints_per_frame_med") or 0)
    low_conf_block = bool(
        conf_mean is not None
        and math.isfinite(float(conf_mean))
        and float(conf_mean) < float(live_guard["min_conf_mean"])
    )
    low_joints_block = bool(joints_med > 0 and joints_med < int(live_guard["min_joints_med"]))
    occlusion_block = bool(low_conf_block or low_joints_block)
    low_motion_high_conf_bypass = False
    _mark_perf("specs_and_guards_ms")

    # Soft stale-drop guard: only drop severely stale windows when current window is low risk.
    seq_in: Optional[int] = None
    try:
        if window_seq is not None:
            seq_in = int(window_seq)
    except (TypeError, ValueError):
        seq_in = None
    seq_prev: Optional[int] = None
    try:
        if st.get("last_window_seq") is not None:
            seq_prev = int(st.get("last_window_seq"))
    except (TypeError, ValueError):
        seq_prev = None

    stale_drop = False
    stale_reason = None
    if bool(live_guard["enable_stale_drop"]) and seq_in is not None and seq_prev is not None:
        lag = int(seq_prev - seq_in)
        severe_stale = bool(lag >= 2)
        low_risk_window = bool(low_motion_block or structural_quality_block or low_quality_block)
        if severe_stale and low_risk_window:
            stale_drop = True
            stale_reason = f"seq_lag={lag},low_risk=1"

    if stale_drop:
        tri_prev = str(st.get("last_triage_state") or "not_fall")
        st["last_window_seq"] = seq_in
        st["last_triage_state"] = tri_prev
        latency_ms = int((time.time() - t0) * 1000)
        perf["total_ms"] = int((time.perf_counter() - perf_started) * 1000)
        resp = build_stale_monitor_response(
            triage_state=tri_prev,
            latency_ms=latency_ms,
            cap_fps_est=cap_fps_est,
            expected_fps=expected_fps,
            effective_fps=float(effective_fps),
            target_T=target_T,
            motion_score=motion_score,
            recent_motion_support=recent_motion_support,
            low_motion_high_conf_bypass=low_motion_high_conf_bypass,
            min_motion=float(min_motion),
            low_motion_block=low_motion_block,
            low_quality_block=low_quality_block,
            structural_quality_block=structural_quality_block,
            occlusion_block=occlusion_block,
            low_conf_block=low_conf_block,
            low_joints_block=low_joints_block,
            live_guard=live_guard,
            qdiag={**qdiag, "sampling_mode": sampling_mode, "low_fps_mode": low_fps_mode},
            raw_stats=raw_stats,
            dataset_code=dataset_code,
            requested_mode=requested_mode,
            mode=mode,
            op_code=op_code,
            effective_use_mc=bool(effective_use_mc),
        requested_use_mc=bool(requested_use_mc),
        effective_mc_M=int(effective_mc_M),
        stale_reason=stale_reason,
        window_end_t_ms=float(window_end_t_ms) if window_end_t_ms is not None else None,
        seq_in=seq_in,
        seq_prev=seq_prev,
    )
        log_monitor_perf_if_slow(
            logger,
            latency_ms=latency_ms,
            session_id=session_id,
            input_source=input_source,
            mode=mode,
            dataset_code=dataset_code,
            op_code=op_code,
            seq_in=seq_in,
            is_replay=is_replay,
            persisted=False,
            perf=perf,
            stale_drop=True,
        )
        return _compact_monitor_response(resp, mode) if compact_response else resp

    def _guard_alert_p(p_raw: float, tau_low: float) -> float:
        """Clamp tracker input when live sampling/motion quality is insufficient."""
        if (
            (
                bool(live_guard["enable_low_motion_gate"])
                and low_motion_block
                and not recent_motion_support
                and not low_motion_high_conf_bypass
            )
            or (bool(live_guard["enable_structural_gate"]) and structural_quality_block)
        ):
            return float(min(float(p_raw), float(tau_low) - 0.02))
        return float(p_raw)

    # -------------
    # Inference
    # -------------

    models_out: Dict[str, Any] = {}
    tri_tcn = None
    tri_gcn = None
    dual_policy_alerts: Dict[str, Any] = {}
    low_motion_high_conf_bypass = False

    run_tcn = mode in {"tcn", "hybrid"}
    run_gcn = mode in {"gcn", "hybrid"}

    if run_tcn:
        t_inf = time.perf_counter()
        out_tcn = _predict_spec(
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
        bypass_tcn = _low_motion_high_conf_bypass(
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
        p_alert_tcn = _guard_alert_p(p_raw_tcn, tau_low_tcn)
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
            trk = OnlineAlertTracker(cfg_tcn)
            st_trackers[tcn_key] = trk
            st_trackers_cfg[tcn_key] = cfg_tcn
        r = trk.step(
            p=float(p_alert_tcn),
            t_s=_t_s,
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
        perf["infer_tcn_ms"] = int((time.perf_counter() - t_inf) * 1000)

        # Optional dual-policy overlays (safe/recall) for deployment UI.
        for pol in ("safe", "recall"):
            pol_cfg = _load_dual_policy_cfg(dataset_code, pol, op_code)
            if not isinstance(pol_cfg, dict):
                continue
            pol_key = f"{tcn_key}::dual::{pol}"
            pol_trk = st_trackers.get(pol_key)
            if pol_trk is None or st_trackers_cfg.get(pol_key) != pol_cfg:
                pol_trk = OnlineAlertTracker(pol_cfg)
                st_trackers[pol_key] = pol_trk
                st_trackers_cfg[pol_key] = pol_cfg

            pol_res = pol_trk.step(
                p=float(p_alert_tcn),
                t_s=_t_s,
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
        t_inf = time.perf_counter()
        out_gcn = _predict_spec(
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
        bypass_gcn = _low_motion_high_conf_bypass(
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
        p_alert_gcn = _guard_alert_p(p_raw_gcn, tau_low_gcn)
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
            trk = OnlineAlertTracker(cfg_gcn)
            st_trackers[gcn_key] = trk
            st_trackers_cfg[gcn_key] = cfg_gcn
        res = trk.step(p=float(p_alert_gcn), t_s=_t_s)
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
        perf["infer_gcn_ms"] = int((time.perf_counter() - t_inf) * 1000)
    _mark_perf("post_infer_policy_ms")

    # Primary alert signals.
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
        # Prefer safety channel state when available to reduce spurious live false alerts.
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
        # Safety hybrid fusion:
        # - automated alert channel (safe): TCN_safe AND GCN_fall
        # - review channel (recall): TCN_recall OR GCN_fall
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
        if recall_alert is not None:
            recall_alert = False

    prev_triage_state = str(st.get("last_triage_state") or "not_fall")
    saved_event_id = None
    notification_dispatch: Optional[Dict[str, Any]] = None

    if mode == "tcn":
        # Use safety-channel alert gate if present (deployment-safe default).
        if "safe" in dual_policy_alerts:
            started_event = bool(dual_policy_alerts.get("safe", {}).get("started_event"))
        else:
            started_event = started_tcn
    elif mode == "gcn":
        started_event = started_gcn
    else:
        # Hybrid safe alert may persist across many windows; persist only on rising edge.
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

    # In low-fps mode, allow falls but require stricter persistence + motion + structure quality.
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
        if recall_alert is not None:
            recall_alert = False
    if bool(live_guard["enable_occlusion_gate"]) and occlusion_block and triage_state == "fall":
        triage_state = "uncertain"
        started_event = False
        safe_alert = False
        if recall_alert is not None:
            recall_alert = False
    if bool(live_guard["enable_structural_gate"]) and structural_quality_block and triage_state == "fall":
        triage_state = "uncertain"
        started_event = False
        safe_alert = False
        if recall_alert is not None:
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
                if recall_alert is not None:
                    recall_alert = False
        else:
            low_fps_confirm_count = 0
            st[low_fps_gate_key] = 0
            low_fps_gate_reason = "failed_low_fps_strict_gate"
            triage_state = "uncertain"
            started_event = False
            safe_alert = False
            if recall_alert is not None:
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
        gate_state_key = f"delivery_gate:{primary_spec_key}:{_norm_op_code(op_code)}"
        if triage_state == "fall":
            gate_state = st.setdefault(gate_state_key, {})
            if started_event or not bool(gate_state.get("active", False)):
                gate_state.clear()
                gate_state["active"] = True
                gate_state["event_start_t_s"] = float(_t_s)
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
            event_start_s = float(gate_state.get("event_start_t_s", _t_s)) - float(st.get("session_start_t_s", _t_s))
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

    # Resolve channel states consistently for all modes.
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

    # Persistence/notification dedup:
    # - only save on event start (started_event)
    # - enforce cooldown so one continuous fall segment produces one event/notification.
    if mode == "tcn":
        cooldown_s_for_persist = float((models_out.get("tcn", {}).get("alert_cfg") or {}).get("cooldown_s", 30.0))
    elif mode == "gcn":
        cooldown_s_for_persist = float((models_out.get("gcn", {}).get("alert_cfg") or {}).get("cooldown_s", 30.0))
    else:
        # Hybrid: use safe-channel cooldown when present, fallback to max model cooldown.
        safe_pol = dual_policy_alerts.get("safe", {}) if isinstance(dual_policy_alerts.get("safe"), dict) else {}
        c_tcn = float((models_out.get("tcn", {}).get("alert_cfg") or {}).get("cooldown_s", 30.0))
        c_gcn = float((models_out.get("gcn", {}).get("alert_cfg") or {}).get("cooldown_s", 30.0))
        cooldown_s_for_persist = float(safe_pol.get("cooldown_s", max(c_tcn, c_gcn)))
    cooldown_s_for_persist = max(1.0, cooldown_s_for_persist)

    persistence = resolve_monitor_persistence_plan(
        session_state=st,
        resident_id=resident_id,
        dataset_code=dataset_code,
        mode=mode,
        op_code=op_code,
        current_t_s=_t_s,
        cooldown_s=cooldown_s_for_persist,
        is_replay=is_replay,
        persist=persist,
        triage_state=str(triage_state),
        prev_triage_state=str(prev_triage_state),
        started_event=bool(started_event),
    )
    persist_event_type = persistence.persist_event_type
    persist_in_cooldown = persistence.persist_in_cooldown
    persist_dedup_key = persistence.persist_dedup_key
    persist_dedup_hits = persistence.persist_dedup_hits
    persist_suppressed = persistence.persist_suppressed
    last_persist_ts = persistence.last_persist_ts
    cooldown_s_for_persist = persistence.cooldown_s

    if persist and persist_event_type and (not persist_in_cooldown):
        t_persist = time.perf_counter()
        primary_out = models_out.get(primary_model_key, {}) if isinstance(models_out.get(primary_model_key), dict) else {}
        safe_guard_probability = float(primary_out.get("mu", p_display) or p_display)
        safe_guard_uncertainty = float(primary_out.get("sigma", 0.0) or 0.0)
        triage_cfg = primary_out.get("triage", {}) if isinstance(primary_out.get("triage"), dict) else {}
        safe_guard_threshold = float(
            triage_cfg.get(
                "tau_high",
                primary_out.get("tau_high", 0.0),
            )
            or 0.0
        )
        safe_guard_margin = float(safe_guard_probability - safe_guard_threshold)
        try:
            with get_conn() as conn:
                saved_event_id, notification_dispatch = persist_monitor_event(
                    conn,
                    session_state=st,
                    persistence=persistence,
                    current_t_s=_t_s,
                    resident_id=resident_id,
                    event_location=event_location,
                    input_source=input_source,
                    requested_mode=requested_mode,
                    effective_mode=mode,
                    runtime=runtime,
                    table_exists=_table_exists,
                    triage_state=str(triage_state),
                    safe_alert=bool(safe_alert),
                    safe_state_out=str(safe_state_out),
                    recall_alert=bool(recall_alert),
                    recall_state_out=str(recall_state_out),
                    expected_fps=float(expected_fps),
                    capture_fps_est=cap_fps_est,
                    p_display=float(p_display),
                    primary_probability=float(safe_guard_probability),
                    primary_uncertainty=float(safe_guard_uncertainty),
                    primary_threshold=float(safe_guard_threshold),
                    models_out=models_out,
                    dual_policy_alerts=dual_policy_alerts,
                )
        except (MySQLError, RuntimeError, OSError, TypeError, ValueError) as exc:
            logger.warning(
                "monitor.predict_window: failed to persist event (resident_id=%s, session_id=%s, mode=%s, dataset=%s): %s",
                resident_id,
                session_id,
                mode,
                dataset_code,
                exc,
            )
        perf["persist_ms"] = int((time.perf_counter() - t_persist) * 1000)

    latency_ms = int((time.time() - t0) * 1000)
    perf["total_ms"] = int((time.perf_counter() - perf_started) * 1000)
    set_last_prediction_snapshot(
        latency_ms=latency_ms,
        p_fall=float(p_display),
        decision=str(triage_state),
        model_code=str(active_model_code),
        ts_iso=datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    )
    st["last_window_seq"] = seq_in
    st["last_triage_state"] = str(triage_state)

    resp = build_monitor_prediction_response(
        triage_state=triage_state,
        models_out=models_out,
        dual_policy_alerts=dual_policy_alerts,
        safe_alert=bool(safe_alert),
        safe_state_out=str(safe_state_out),
        recall_alert=bool(recall_alert),
        recall_state_out=str(recall_state_out),
        latency_ms=latency_ms,
        cap_fps_est=cap_fps_est,
        expected_fps=expected_fps,
        effective_fps=float(effective_fps),
        target_T=target_T,
        motion_score=motion_score,
        lying_score=lying_score,
        confirm_motion_score=confirm_motion_score,
        low_motion_block=low_motion_block,
        recent_motion_support=recent_motion_support,
        low_motion_high_conf_bypass=low_motion_high_conf_bypass,
        min_motion=float(min_motion),
        low_quality_block=low_quality_block,
        structural_quality_block=structural_quality_block,
        occlusion_block=occlusion_block,
        low_conf_block=low_conf_block,
        low_joints_block=low_joints_block,
        live_guard=live_guard,
        sampling_mode=sampling_mode,
        low_fps_mode=low_fps_mode,
        low_fps_confirm_count=int(low_fps_confirm_count),
        low_fps_need=int(low_fps_need),
        low_fps_gate_reason=low_fps_gate_reason,
        qdiag=qdiag,
        raw_stats=raw_stats,
        dataset_code=dataset_code,
        requested_mode=requested_mode,
        mode=mode,
        op_code=op_code,
        effective_use_mc=bool(effective_use_mc),
        requested_use_mc=bool(requested_use_mc),
        effective_mc_M=int(effective_mc_M),
        delivery_gate_diag=delivery_gate_diag,
        uncertain_promoted=bool(uncertain_promoted),
        saved_event_id=saved_event_id,
        notification_dispatch=notification_dispatch,
        persist_suppressed=persist_suppressed,
        cooldown_s_for_persist=float(cooldown_s_for_persist),
        persist_dedup_hits=int(persist_dedup_hits),
        current_t_s=_t_s,
        last_persist_ts=last_persist_ts,
        persist_dedup_key=persist_dedup_key,
        session_state=st,
        window_end_t_ms=float(window_end_t_ms) if window_end_t_ms is not None else None,
        seq_in=seq_in,
        seq_prev=seq_prev,
    )
    log_monitor_perf_if_slow(
        logger,
        latency_ms=latency_ms,
        session_id=session_id,
        input_source=input_source,
        mode=mode,
        dataset_code=dataset_code,
        op_code=op_code,
        seq_in=seq_in,
        is_replay=is_replay,
        persisted=bool(saved_event_id is not None or persist_event_type),
        perf=perf,
    )
    return _compact_monitor_response(resp, mode) if compact_response else resp


@router.websocket("/api/monitor/ws")
@router.websocket("/api/v1/monitor/ws")
async def monitor_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            try:
                payload = MonitorPredictPayload.model_validate(data)
                out = predict_window(payload)
                await websocket.send_json(out)
            except HTTPException as exc:
                await websocket.send_json(
                    {
                        "error": True,
                        "detail": exc.detail,
                        "status_code": exc.status_code,
                    }
                )
            except (TypeError, ValueError) as exc:
                await websocket.send_json(
                    {
                        "error": True,
                        "detail": str(exc),
                        "status_code": 400,
                    }
                )
    except WebSocketDisconnect:
        return
