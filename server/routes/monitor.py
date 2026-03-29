from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Body, HTTPException, Query, WebSocket, WebSocketDisconnect
import numpy as np
import yaml

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from .. import core
from ..core import MonitorPredictPayload, _detect_variants, _ensure_system_settings_schema, _table_exists
from ..core import normalize_dataset_code
from ..db import get_conn
from ..deploy_runtime import (
    get_specs as _get_deploy_specs,
    predict_spec as _predict_spec,
)
from ..online_alert import OnlineAlertTracker
from ..services.monitor_context_service import load_monitor_request_context
from ..services.monitor_response_service import (
    build_monitor_prediction_response,
    build_stale_monitor_response,
    log_monitor_perf_if_slow,
)
from ..services.monitor_runtime_service import (
    persist_monitor_event,
    resolve_monitor_persistence_plan,
)
from ..services.monitor_session_service import get_monitor_session_state, reset_monitor_session
from fall_detection.deploy.common import WindowRaw, compute_confirm_scores
from fall_detection.pose.preprocess_pose_npz import (
    linear_fill_small_gaps,
    normalize_body_centric,
    smooth_weighted_moving_average,
    standardize_missing,
)


router = APIRouter()
logger = logging.getLogger(__name__)
_DUAL_POLICY_CFG_CACHE: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
_LIVE_EFFECTIVE_FPS_MIN = 10.0
_DEFAULT_LIVE_GUARD_GLOBAL = {
    "low_fps_mode_threshold": 16.0,
    "low_fps_fall_persist_n": 3,
    "min_frames_ratio": 0.60,
    "min_coverage_ratio": 0.85,
    "min_joints_med": 20,
}
_DEFAULT_LIVE_GUARD_BY_DATASET = {
    "caucafall": {
        "min_motion_for_fall": 0.020,
        "min_fps_ratio": 0.70,
        "min_conf_mean": 0.35,
        "allow_low_motion_high_conf_bypass": False,
        "low_motion_high_conf_k": 0,
        "low_motion_high_conf_max_lying": None,
    },
    "le2i": {
        "min_motion_for_fall": 0.020,
        "min_fps_ratio": 0.70,
        "min_conf_mean": 0.35,
        "allow_low_motion_high_conf_bypass": False,
        "low_motion_high_conf_k": 0,
        "low_motion_high_conf_max_lying": None,
    },
}
_MONITOR_Q_SCALE = 1000.0
_LOW_MOTION_MEMORY_WINDOWS = 5
_ONLINE_RAW_PREPROC = {
    "conf_thr": 0.2,
    "smooth_k": 5,
    "max_gap": 4,
    "fill_conf": "thr",
    "normalize": "torso",
    "pelvis_fill": "nearest",
    "rotate": "none",
}


def _norm_op_code(op_code: str) -> str:
    c = (op_code or "").strip().upper().replace("_", "-")
    if c in {"OP1", "OP-1", "1"}:
        return "OP-1"
    if c in {"OP2", "OP-2", "2"}:
        return "OP-2"
    if c in {"OP3", "OP-3", "3"}:
        return "OP-3"
    return "OP-2"


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


def _op_live_guard(specs: Dict[str, Any], spec_key: str, op_code: str, dataset_code: str) -> Dict[str, Any]:
    """Resolve live guard config from ops.<OP*>.live_guard with sane defaults."""
    ds_defaults = _DEFAULT_LIVE_GUARD_BY_DATASET.get(dataset_code, {})
    d_min_motion = float(ds_defaults.get("min_motion_for_fall", 0.020))
    d_low_fps_thr = float(_DEFAULT_LIVE_GUARD_GLOBAL.get("low_fps_mode_threshold", 16.0))
    d_low_fps_persist_n = int(_DEFAULT_LIVE_GUARD_GLOBAL.get("low_fps_fall_persist_n", 3))
    d_min_fps_ratio = float(ds_defaults.get("min_fps_ratio", 0.70))
    d_min_frames_ratio = float(_DEFAULT_LIVE_GUARD_GLOBAL.get("min_frames_ratio", 0.60))
    d_min_coverage_ratio = float(_DEFAULT_LIVE_GUARD_GLOBAL.get("min_coverage_ratio", 0.85))
    d_min_conf_mean = float(ds_defaults.get("min_conf_mean", 0.35))
    d_min_joints_med = int(_DEFAULT_LIVE_GUARD_GLOBAL.get("min_joints_med", 20))
    out = {
        "min_motion_for_fall": d_min_motion,
        "low_fps_mode_threshold": d_low_fps_thr,
        "low_fps_fall_persist_n": d_low_fps_persist_n,
        "min_fps_ratio": d_min_fps_ratio,
        "min_frames_ratio": d_min_frames_ratio,
        "min_coverage_ratio": d_min_coverage_ratio,
        "min_conf_mean": d_min_conf_mean,
        "min_joints_med": d_min_joints_med,
        "enable_stale_drop": True,
        "enable_low_motion_gate": True,
        "enable_occlusion_gate": True,
        "enable_structural_gate": True,
        "enable_low_fps_persist_gate": True,
        "allow_low_motion_high_conf_bypass": _coerce_bool(ds_defaults.get("allow_low_motion_high_conf_bypass"), False),
        "low_motion_high_conf_k": int(ds_defaults.get("low_motion_high_conf_k", 0) or 0),
        "low_motion_high_conf_max_lying": ds_defaults.get("low_motion_high_conf_max_lying"),
    }
    try:
        spec = specs.get(spec_key)
        ops = spec.ops if spec is not None and hasattr(spec, "ops") else {}
        op = (ops or {}).get(_norm_op_code(op_code)) or {}
        lg = op.get("live_guard") if isinstance(op, dict) else {}
        if isinstance(lg, dict):
            out["min_motion_for_fall"] = float(lg.get("min_motion_for_fall", out["min_motion_for_fall"]))
            out["low_fps_mode_threshold"] = float(lg.get("low_fps_mode_threshold", out["low_fps_mode_threshold"]))
            out["low_fps_fall_persist_n"] = int(lg.get("low_fps_fall_persist_n", out["low_fps_fall_persist_n"]))
            out["min_fps_ratio"] = float(lg.get("min_fps_ratio", out["min_fps_ratio"]))
            out["min_frames_ratio"] = float(lg.get("min_frames_ratio", out["min_frames_ratio"]))
            out["min_coverage_ratio"] = float(lg.get("min_coverage_ratio", out["min_coverage_ratio"]))
            out["min_conf_mean"] = float(lg.get("min_conf_mean", out["min_conf_mean"]))
            out["min_joints_med"] = int(lg.get("min_joints_med", out["min_joints_med"]))
            out["enable_stale_drop"] = _coerce_bool(lg.get("enable_stale_drop"), out["enable_stale_drop"])
            out["enable_low_motion_gate"] = _coerce_bool(lg.get("enable_low_motion_gate"), out["enable_low_motion_gate"])
            out["enable_occlusion_gate"] = _coerce_bool(lg.get("enable_occlusion_gate"), out["enable_occlusion_gate"])
            out["enable_structural_gate"] = _coerce_bool(lg.get("enable_structural_gate"), out["enable_structural_gate"])
            out["enable_low_fps_persist_gate"] = _coerce_bool(
                lg.get("enable_low_fps_persist_gate"), out["enable_low_fps_persist_gate"]
            )
            out["allow_low_motion_high_conf_bypass"] = _coerce_bool(
                lg.get("allow_low_motion_high_conf_bypass"), out["allow_low_motion_high_conf_bypass"]
            )
            out["low_motion_high_conf_k"] = int(lg.get("low_motion_high_conf_k", out["low_motion_high_conf_k"]))
            if lg.get("low_motion_high_conf_max_lying") is not None:
                out["low_motion_high_conf_max_lying"] = float(lg.get("low_motion_high_conf_max_lying"))
    except (TypeError, ValueError, AttributeError):
        pass
    # Clamp to safe bounds.
    out["min_motion_for_fall"] = float(max(0.0, out["min_motion_for_fall"]))
    out["low_fps_mode_threshold"] = float(max(5.0, out["low_fps_mode_threshold"]))
    out["low_fps_fall_persist_n"] = int(max(1, out["low_fps_fall_persist_n"]))
    out["min_fps_ratio"] = float(min(1.5, max(0.1, out["min_fps_ratio"])))
    out["min_frames_ratio"] = float(min(1.0, max(0.1, out["min_frames_ratio"])))
    out["min_coverage_ratio"] = float(min(1.2, max(0.1, out["min_coverage_ratio"])))
    out["min_conf_mean"] = float(min(1.0, max(0.0, out["min_conf_mean"])))
    out["min_joints_med"] = int(max(1, out["min_joints_med"]))
    out["low_motion_high_conf_k"] = int(max(0, out["low_motion_high_conf_k"]))
    return out


def _op_delivery_gate(specs: Dict[str, Any], spec_key: str, op_code: str) -> Dict[str, Any]:
    out = {
        "enabled": False,
        "max_lying": None,
        "max_start_lying": None,
        "min_mean_motion_high": None,
        "max_event_start_s": None,
    }
    try:
        spec = specs.get(spec_key)
        ops = spec.ops if spec is not None and hasattr(spec, "ops") else {}
        op = (ops or {}).get(_norm_op_code(op_code)) or {}
        gate = op.get("delivery_gate") if isinstance(op, dict) else {}
        if isinstance(gate, dict):
            out["enabled"] = _coerce_bool(gate.get("enabled"), False)
            if gate.get("max_lying") is not None:
                out["max_lying"] = float(gate.get("max_lying"))
            if gate.get("max_start_lying") is not None:
                out["max_start_lying"] = float(gate.get("max_start_lying"))
            if gate.get("min_mean_motion_high") is not None:
                out["min_mean_motion_high"] = float(gate.get("min_mean_motion_high"))
            if gate.get("max_event_start_s") is not None:
                out["max_event_start_s"] = float(gate.get("max_event_start_s"))
    except (TypeError, ValueError, AttributeError):
        pass
    return out


def _op_uncertain_promote(specs: Dict[str, Any], spec_key: str, op_code: str) -> Dict[str, Any]:
    out = {
        "enabled": False,
        "video_only": True,
        "min_p_alert": None,
        "min_motion": None,
        "max_lying": None,
    }
    try:
        spec = specs.get(spec_key)
        ops = spec.ops if spec is not None and hasattr(spec, "ops") else {}
        op = (ops or {}).get(_norm_op_code(op_code)) or {}
        cfg = op.get("uncertain_promote") if isinstance(op, dict) else {}
        if isinstance(cfg, dict):
            out["enabled"] = _coerce_bool(cfg.get("enabled"), False)
            out["video_only"] = _coerce_bool(cfg.get("video_only"), True)
            if cfg.get("min_p_alert") is not None:
                out["min_p_alert"] = float(cfg.get("min_p_alert"))
            if cfg.get("min_motion") is not None:
                out["min_motion"] = float(cfg.get("min_motion"))
            if cfg.get("max_lying") is not None:
                out["max_lying"] = float(cfg.get("max_lying"))
    except (TypeError, ValueError, AttributeError):
        pass
    return out


def _load_dual_policy_cfg(dataset_code: str, policy_name: str, op_code: str) -> Optional[Dict[str, Any]]:
    """Load alert cfg for dual policy overlays (safe/recall) if available."""
    key = (f"{dataset_code}:{policy_name}", _norm_op_code(op_code))
    if key in _DUAL_POLICY_CFG_CACHE:
        return _DUAL_POLICY_CFG_CACHE[key]

    root = Path(__file__).resolve().parents[2]
    path = root / "configs" / "ops" / "dual_policy" / f"tcn_{dataset_code}_dual_{policy_name}.yaml"
    if not path.exists():
        _DUAL_POLICY_CFG_CACHE[key] = None
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        _DUAL_POLICY_CFG_CACHE[key] = None
        return None

    if not isinstance(data, dict):
        _DUAL_POLICY_CFG_CACHE[key] = None
        return None

    cfg = {}
    if isinstance(data.get("alert_cfg"), dict):
        cfg.update(data.get("alert_cfg") or {})
    elif isinstance(data.get("alert_base"), dict):
        cfg.update(data.get("alert_base") or {})

    ops = data.get("ops") if isinstance(data.get("ops"), dict) else {}
    op_entry = None
    want = _norm_op_code(op_code)
    for k, v in (ops or {}).items():
        if not isinstance(v, dict):
            continue
        kk = str(k).strip().upper().replace("_", "-")
        if kk in {want, want.replace("-", "")}:
            op_entry = v
            break
    if op_entry is None and isinstance(ops, dict):
        op_entry = (ops.get("OP2") or ops.get("OP-2") or ops.get("op2") or ops.get("op-2"))

    if isinstance(op_entry, dict):
        if op_entry.get("tau_low") is not None:
            cfg["tau_low"] = float(op_entry.get("tau_low"))
        if op_entry.get("tau_high") is not None:
            cfg["tau_high"] = float(op_entry.get("tau_high"))

    # Minimal required defaults for OnlineAlertTracker.
    cfg.setdefault("ema_alpha", 0.2)
    cfg.setdefault("k", 2)
    cfg.setdefault("n", 3)
    cfg.setdefault("cooldown_s", 30.0)
    cfg.setdefault("tau_low", 0.5)
    cfg.setdefault("tau_high", 0.85)

    _DUAL_POLICY_CFG_CACHE[key] = cfg
    return cfg

def _window_motion_score(xy: Any) -> Optional[float]:
    """Robust per-window motion score on normalized XY coordinates.

    Uses torso-center displacement (shoulders + hips) and normalizes by torso size
    to reduce both jitter and camera distance scale effects.
    Returns median normalized displacement per frame.
    """
    if not isinstance(xy, list) or len(xy) < 2:
        return None

    def _torso_center_scale(frame: Any) -> Optional[Tuple[float, float, float]]:
        if not isinstance(frame, list):
            return None
        pts: Dict[int, Tuple[float, float]] = {}
        # MediaPipe Pose torso anchors: shoulders + hips
        for idx in (11, 12, 23, 24):
            if idx >= len(frame):
                continue
            p = frame[idx]
            if not isinstance(p, list) or len(p) < 2:
                continue
            try:
                x = float(p[0])
                y = float(p[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                pts[idx] = (x, y)
        if len(pts) < 2:
            return None
        vals = list(pts.values())
        sx = sum(p[0] for p in vals) / len(vals)
        sy = sum(p[1] for p in vals) / len(vals)

        # Torso scale estimate: shoulder width / hip width / shoulder-hip vertical span.
        scales: List[float] = []
        if 11 in pts and 12 in pts:
            dx = pts[12][0] - pts[11][0]
            dy = pts[12][1] - pts[11][1]
            scales.append(math.hypot(dx, dy))
        if 23 in pts and 24 in pts:
            dx = pts[24][0] - pts[23][0]
            dy = pts[24][1] - pts[23][1]
            scales.append(math.hypot(dx, dy))
        if 11 in pts and 23 in pts:
            dx = pts[23][0] - pts[11][0]
            dy = pts[23][1] - pts[11][1]
            scales.append(math.hypot(dx, dy))
        if 12 in pts and 24 in pts:
            dx = pts[24][0] - pts[12][0]
            dy = pts[24][1] - pts[12][1]
            scales.append(math.hypot(dx, dy))

        scale = max(1e-6, (sum(scales) / len(scales)) if scales else 0.08)
        return (sx, sy, scale)

    speeds: List[float] = []
    for i in range(1, len(xy)):
        c0 = _torso_center_scale(xy[i - 1])
        c1 = _torso_center_scale(xy[i])
        if c0 is None or c1 is None:
            continue
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        if math.isfinite(dx) and math.isfinite(dy):
            disp = math.hypot(dx, dy)
            s = max(1e-6, 0.5 * (c0[2] + c1[2]))
            speeds.append(disp / s)

    if not speeds:
        return None
    speeds.sort()
    m = len(speeds) // 2
    if len(speeds) % 2 == 1:
        return float(speeds[m])
    return float((speeds[m - 1] + speeds[m]) * 0.5)


def _raw_window_stats(raw_t_ms: Any, raw_xy: Any, raw_conf: Any) -> Dict[str, Any]:
    """Compute lightweight payload diagnostics for monitor parity checks."""
    stats: Dict[str, Any] = {
        "raw_len": 0,
        "raw_fps_est": None,
        "raw_duration_s": None,
        "joints_per_frame_med": None,
        "conf_mean": None,
    }
    if not isinstance(raw_t_ms, list) or not isinstance(raw_xy, list):
        return stats
    n = min(len(raw_t_ms), len(raw_xy))
    if n <= 0:
        return stats
    stats["raw_len"] = int(n)

    t_vals: List[float] = []
    joints_counts: List[int] = []
    conf_vals: List[float] = []
    for i in range(n):
        try:
            t_vals.append(float(raw_t_ms[i]))
        except (TypeError, ValueError):
            continue
        frame = raw_xy[i]
        if isinstance(frame, list):
            joints_counts.append(len(frame))
        cfr = raw_conf[i] if isinstance(raw_conf, list) and i < len(raw_conf) else None
        if isinstance(cfr, list):
            for v in cfr:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    conf_vals.append(fv)

    if len(t_vals) >= 2:
        dt_s = (t_vals[-1] - t_vals[0]) / 1000.0
        if dt_s > 1e-6:
            stats["raw_duration_s"] = float(dt_s)
            stats["raw_fps_est"] = float((len(t_vals) - 1) / dt_s)

    if joints_counts:
        joints_counts.sort()
        m = len(joints_counts) // 2
        stats["joints_per_frame_med"] = int(joints_counts[m])

    if conf_vals:
        stats["conf_mean"] = float(sum(conf_vals) / max(1, len(conf_vals)))
    return stats


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
        "stale_drop": resp.get("stale_drop"),
        "stale_reason": resp.get("stale_reason"),
        "models": compact_models,
        "policy_alerts": _compact_policy_alerts(resp.get("policy_alerts")),
    }


def _direct_window_stats(xy: Any, conf: Any, *, effective_fps: float) -> Dict[str, Any]:
    """Fallback stats for already-resampled direct windows.

    When the caller sends fixed-length `xy/conf` directly, there is no raw timing
    stream to derive coverage metrics from. Treat the payload as a complete window
    at `effective_fps` so live quality gates do not misclassify it as structurally
    incomplete.
    """
    stats: Dict[str, Any] = {
        "raw_len": 0,
        "raw_fps_est": None,
        "raw_duration_s": None,
        "joints_per_frame_med": None,
        "conf_mean": None,
    }
    if not isinstance(xy, list) or len(xy) <= 0:
        return stats

    n = len(xy)
    stats["raw_len"] = int(n)
    fps_eff = float(effective_fps) if effective_fps and float(effective_fps) > 0 else None
    if fps_eff is not None:
        stats["raw_fps_est"] = fps_eff
        if n >= 2:
            stats["raw_duration_s"] = float((n - 1) / fps_eff)
        else:
            stats["raw_duration_s"] = 0.0

    joint_counts: List[int] = []
    conf_vals: List[float] = []
    for i, frame in enumerate(xy):
        if isinstance(frame, list):
            joint_counts.append(len(frame))
        cfr = conf[i] if isinstance(conf, list) and i < len(conf) else None
        if isinstance(cfr, list):
            for v in cfr:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    conf_vals.append(fv)

    if joint_counts:
        joint_counts.sort()
        stats["joints_per_frame_med"] = int(joint_counts[len(joint_counts) // 2])
    if conf_vals:
        stats["conf_mean"] = float(sum(conf_vals) / max(1, len(conf_vals)))
    return stats


def _decode_quantized_raw_window(
    raw_xy_q: Any,
    raw_conf_q: Any,
    raw_shape: Any,
) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """Decode compact quantized window payload into nested xy/conf arrays."""
    try:
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) < 2:
            return None, None
        n = int(raw_shape[0])
        j = int(raw_shape[1])
        if n <= 0 or j <= 0:
            return None, None
    except (TypeError, ValueError):
        return None, None

    if not isinstance(raw_xy_q, list) or len(raw_xy_q) != n * j * 2:
        return None, None
    if raw_conf_q is not None and (not isinstance(raw_conf_q, list) or len(raw_conf_q) != n * j):
        return None, None

    xy_out: List[Any] = []
    conf_out: List[Any] = []
    qi = 0
    ci = 0
    for _ in range(n):
        fxy: List[Any] = []
        fconf: List[Any] = []
        for _j in range(j):
            try:
                x = float(raw_xy_q[qi]) / _MONITOR_Q_SCALE
                y = float(raw_xy_q[qi + 1]) / _MONITOR_Q_SCALE
            except (TypeError, ValueError, IndexError):
                x = 0.0
                y = 0.0
            qi += 2
            fxy.append([x, y])
            if raw_conf_q is not None:
                try:
                    c = float(raw_conf_q[ci]) / _MONITOR_Q_SCALE
                except (TypeError, ValueError, IndexError):
                    c = 0.0
                ci += 1
            else:
                c = 1.0
            fconf.append(c)
        xy_out.append(fxy)
        conf_out.append(fconf)
    return xy_out, conf_out


def _window_quality_block(
    *,
    raw_stats: Dict[str, Any],
    expected_fps: float,
    effective_fps: float,
    target_T: int,
    dataset_code: str,
    min_fps_ratio_override: Optional[float] = None,
    min_frames_ratio_override: Optional[float] = None,
    min_coverage_ratio_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Return quality diagnostics + whether this window should be alert-blocked."""
    need_s = max(1e-6, (max(2, int(target_T)) - 1) / max(1e-6, float(effective_fps)))
    raw_fps = raw_stats.get("raw_fps_est")
    raw_len = int(raw_stats.get("raw_len") or 0)
    raw_dur = raw_stats.get("raw_duration_s")

    fps_ratio = (float(raw_fps) / float(expected_fps)) if raw_fps is not None and expected_fps > 0 else None
    frame_ratio = (float(raw_len) / float(target_T)) if target_T > 0 else None
    coverage_ratio = (float(raw_dur) / float(need_s)) if raw_dur is not None and need_s > 0 else None

    ds_defaults = _DEFAULT_LIVE_GUARD_BY_DATASET.get(dataset_code, {})
    min_fps_ratio = float(
        min_fps_ratio_override
        if min_fps_ratio_override is not None
        else ds_defaults.get("min_fps_ratio", 0.75)
    )
    min_frames_ratio = float(
        min_frames_ratio_override
        if min_frames_ratio_override is not None
        else _DEFAULT_LIVE_GUARD_GLOBAL.get("min_frames_ratio", 0.60)
    )
    min_coverage_ratio = float(
        min_coverage_ratio_override
        if min_coverage_ratio_override is not None
        else _DEFAULT_LIVE_GUARD_GLOBAL.get("min_coverage_ratio", 0.85)
    )
    low_fps = bool(fps_ratio is not None and fps_ratio < min_fps_ratio)
    low_frames = bool(frame_ratio is not None and frame_ratio < min_frames_ratio)
    low_coverage = bool(coverage_ratio is not None and coverage_ratio < min_coverage_ratio)

    return {
        "need_duration_s": float(need_s),
        "fps_ratio": float(fps_ratio) if fps_ratio is not None else None,
        "frame_ratio": float(frame_ratio) if frame_ratio is not None else None,
        "coverage_ratio": float(coverage_ratio) if coverage_ratio is not None else None,
        "min_fps_ratio": float(min_fps_ratio),
        "min_frames_ratio": float(min_frames_ratio),
        "min_coverage_ratio": float(min_coverage_ratio),
        "low_fps": low_fps,
        "low_frames": low_frames,
        "low_coverage": low_coverage,
        "low_quality_block": bool(low_fps or low_frames or low_coverage),
    }


def _effective_target_fps(*, expected_fps: float, raw_fps_est: Optional[float]) -> float:
    """Choose runtime target fps for resampling.

    We clamp to a safe range so low frontend fps can still be consumed without
    over-upsampling to an unrealistic rate.
    """
    exp = float(expected_fps) if expected_fps and float(expected_fps) > 0 else 23.0
    if raw_fps_est is None or not math.isfinite(float(raw_fps_est)) or float(raw_fps_est) <= 0:
        return exp
    raw = float(raw_fps_est)
    lo = min(_LIVE_EFFECTIVE_FPS_MIN, exp)
    hi = exp
    return float(max(lo, min(raw, hi)))


def _preprocess_online_raw_window(xy: Any, conf: Any) -> Tuple[List[Any], List[Any]]:
    """Match online raw inference to the same pose-preprocessed domain used offline."""
    if not isinstance(xy, list) or len(xy) <= 0:
        return [], []

    xy_np = np.asarray(xy, dtype=np.float32)
    conf_np = np.asarray(conf, dtype=np.float32) if isinstance(conf, list) and len(conf) == len(xy) else None
    if conf_np is None:
        conf_np = np.ones((xy_np.shape[0], xy_np.shape[1]), dtype=np.float32)

    if xy_np.ndim != 3 or xy_np.shape[-1] != 2 or conf_np.ndim != 2:
        return xy, conf if isinstance(conf, list) else []

    cfg = _ONLINE_RAW_PREPROC
    xy_np, conf_np = standardize_missing(xy_np, conf_np)
    xy_np, conf_np, _ = linear_fill_small_gaps(
        xy_np,
        conf_np,
        conf_thr=float(cfg["conf_thr"]),
        max_gap=int(cfg["max_gap"]),
        fill_conf=str(cfg["fill_conf"]),
    )
    xy_np = smooth_weighted_moving_average(
        xy_np,
        conf_np,
        conf_thr=float(cfg["conf_thr"]),
        k=int(cfg["smooth_k"]),
    )
    xy_np, _ = normalize_body_centric(
        xy_np,
        conf_np,
        conf_thr=float(cfg["conf_thr"]),
        mode=str(cfg["normalize"]),
        pelvis_fill=str(cfg["pelvis_fill"]),
        rotate=str(cfg["rotate"]),
    )
    xy_np = np.nan_to_num(xy_np, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    conf_np = np.nan_to_num(conf_np, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return xy_np.tolist(), conf_np.tolist()



def _resample_pose_window(
    *,
    raw_t_ms: Any,
    raw_xy: Any,
    raw_conf: Any = None,
    target_fps: float = 30.0,
    target_T: int = 48,
    window_end_t_ms: Optional[float] = None,
) -> Tuple[List[Any], List[Any], float, float, Optional[float]]:
    """Resample variable-FPS pose frames to a fixed FPS + fixed length window."""

    if (
        not isinstance(raw_t_ms, list)
        or not isinstance(raw_xy, list)
        or len(raw_t_ms) != len(raw_xy)
        or len(raw_xy) < 1
    ):
        return [], [], 0.0, 0.0, None

    if isinstance(raw_conf, list) and len(raw_conf) == len(raw_xy):
        use_conf = True
    else:
        use_conf = False
        raw_conf = [None] * len(raw_xy)

    t: List[float] = []
    xy: List[Any] = []
    conf: List[Any] = []
    last_t: Optional[float] = None
    for ti, xyi, ci in zip(raw_t_ms, raw_xy, raw_conf):
        try:
            tf = float(ti)
        except (TypeError, ValueError):
            continue
        if last_t is not None and tf <= last_t:
            continue
        if not isinstance(xyi, list):
            continue
        t.append(tf)
        xy.append(xyi)
        conf.append(ci if use_conf else None)
        last_t = tf

    if len(t) < 1:
        return [], [], 0.0, 0.0, None

    cap_fps: Optional[float] = None
    if len(t) >= 2:
        dt_s = (t[-1] - t[0]) / 1000.0
        if dt_s > 1e-6:
            cap_fps = (len(t) - 1) / dt_s

    target_fps = float(target_fps) if target_fps and float(target_fps) > 0 else 30.0
    target_T = int(target_T) if target_T and int(target_T) > 1 else 48
    dt_ms = 1000.0 / target_fps

    end_t_ms = float(window_end_t_ms) if window_end_t_ms is not None else t[-1]
    start_t_ms = end_t_ms - (target_T - 1) * dt_ms

    j = 0
    xy_out: List[Any] = []
    conf_out: List[Any] = []

    def lerp(a: float, b: float, alpha: float) -> float:
        return a + (b - a) * alpha

    def interp_frame(frame0: Any, frame1: Any, alpha: float, is_xy: bool) -> Any:
        if not isinstance(frame0, list):
            return frame1
        if not isinstance(frame1, list):
            return frame0
        out = []
        if is_xy:
            n = min(len(frame0), len(frame1))
            for k in range(n):
                p0 = frame0[k] if isinstance(frame0[k], list) and len(frame0[k]) >= 2 else [0.0, 0.0]
                p1 = frame1[k] if isinstance(frame1[k], list) and len(frame1[k]) >= 2 else [0.0, 0.0]
                try:
                    x0, y0 = float(p0[0]), float(p0[1])
                    x1, y1 = float(p1[0]), float(p1[1])
                except (TypeError, ValueError):
                    x0 = y0 = x1 = y1 = 0.0
                out.append([lerp(x0, x1, alpha), lerp(y0, y1, alpha)])
            if len(frame0) > n:
                out.extend(frame0[n:])
            elif len(frame1) > n:
                out.extend(frame1[n:])
            return out

        n = min(len(frame0), len(frame1))
        for k in range(n):
            try:
                v0 = float(frame0[k])
            except (TypeError, ValueError):
                v0 = 0.0
            try:
                v1 = float(frame1[k])
            except (TypeError, ValueError):
                v1 = v0
            out.append(lerp(v0, v1, alpha))
        if len(frame0) > n:
            out.extend(frame0[n:])
        elif len(frame1) > n:
            out.extend(frame1[n:])
        return out

    for k in range(target_T):
        tw = start_t_ms + k * dt_ms

        if tw <= t[0]:
            xy_out.append(xy[0])
            conf_out.append(
                conf[0]
                if use_conf and isinstance(conf[0], list)
                else [1.0] * (len(xy[0]) if isinstance(xy[0], list) else 0)
            )
            continue
        if tw >= t[-1]:
            xy_out.append(xy[-1])
            conf_out.append(
                conf[-1]
                if use_conf and isinstance(conf[-1], list)
                else [1.0] * (len(xy[-1]) if isinstance(xy[-1], list) else 0)
            )
            continue

        while j + 1 < len(t) and t[j + 1] < tw:
            j += 1

        t0, t1 = t[j], t[j + 1]
        alpha = 0.0 if t1 <= t0 else (tw - t0) / (t1 - t0)

        xy_out.append(interp_frame(xy[j], xy[j + 1], alpha, True))

        if use_conf and isinstance(conf[j], list) and isinstance(conf[j + 1], list):
            conf_out.append(interp_frame(conf[j], conf[j + 1], alpha, False))
        else:
            conf_out.append([1.0] * (len(xy_out[-1]) if isinstance(xy_out[-1], list) else 0))

    return xy_out, conf_out, start_t_ms, end_t_ms, cap_fps


@router.post("/api/monitor/reset_session")
@router.post("/api/v1/monitor/reset_session")
def reset_session(session_id: str = Query(...)) -> Dict[str, Any]:
    return reset_monitor_session(core._SESSION_STATE, session_id)


@router.post("/api/monitor/predict_window")
@router.post("/api/v1/monitor/predict_window")
def predict_window(payload: MonitorPredictPayload = Body(...)) -> Dict[str, Any]:
    """Score one window from the live monitor UI."""
    payload_d = payload.model_dump()
    compact_response = bool(payload_d.get("compact_response"))

    t0 = time.time()
    perf_started = time.perf_counter()
    perf_last = perf_started
    perf: Dict[str, int] = {}

    def _mark_perf(name: str) -> None:
        nonlocal perf_last
        now = time.perf_counter()
        perf[name] = int((now - perf_last) * 1000)
        perf_last = now

    # -------------
    # Inputs
    # -------------
    session_id = str(payload_d.get("session_id") or "default")
    input_source = str(payload_d.get("input_source") or "").strip().lower()
    is_replay = input_source in {"video", "replay", "file"}

    requested_mode = str(payload_d.get("mode") or "tcn").lower().strip()
    mode = requested_mode
    if mode in {"hyb", "hybrid", "dual"}:
        mode = "hybrid"
    elif mode not in {"tcn", "gcn", "hybrid"}:
        mode = "tcn"

    dataset_code = normalize_dataset_code(payload_d.get("dataset_code") or payload_d.get("dataset"))
    op_code = str(payload_d.get("op_code") or payload_d.get("op") or "").upper().strip()

    requested_use_mc = payload_d.get("use_mc")
    requested_mc_M = payload_d.get("mc_M")

    persist = bool(payload_d.get("persist", False))

    target_T = int(payload_d.get("target_T") or 48)
    raw_xy = payload_d.get("raw_xy")
    raw_conf = payload_d.get("raw_conf")
    raw_xy_q = payload_d.get("raw_xy_q")
    raw_conf_q = payload_d.get("raw_conf_q")
    raw_shape = payload_d.get("raw_shape")
    raw_t_ms = payload_d.get("raw_t_ms")
    window_end_t_ms = payload_d.get("window_end_t_ms", None)
    window_seq = payload_d.get("window_seq", None)
    if raw_xy is None and raw_xy_q is not None:
        dec_xy, dec_conf = _decode_quantized_raw_window(raw_xy_q, raw_conf_q, raw_shape)
        if dec_xy is not None:
            raw_xy = dec_xy
            raw_conf = dec_conf

    raw_stats = _raw_window_stats(raw_t_ms, raw_xy, raw_conf)

    xy: List[Any] = []
    conf: List[Any] = []
    cap_fps_est: Optional[float] = None
    _mark_perf("parse_inputs_ms")

    # -------------
    # Defaults from DB (if available)
    # -------------
    resident_id = int(payload_d.get("resident_id") or 1)
    active_model_code = mode.upper()
    try:
        with get_conn() as conn:
            request_context = load_monitor_request_context(
                conn=conn,
                resident_id=resident_id,
                input_source=input_source,
                mode=mode,
                dataset_code=dataset_code,
                op_code=op_code,
                event_location=payload_d.get("location"),
                active_model_code=active_model_code,
                requested_use_mc=requested_use_mc,
                requested_mc_M=requested_mc_M,
                is_replay=is_replay,
                ensure_system_settings_schema=_ensure_system_settings_schema,
                detect_variants=_detect_variants,
                table_exists=_table_exists,
            )
    except (MySQLError, RuntimeError, OSError, TypeError, ValueError) as exc:
        logger.warning(
            "monitor.predict_window: failed to load DB defaults (resident_id=%s, session_id=%s, mode=%s, dataset=%s): %s",
            resident_id,
            session_id,
            mode,
            dataset_code or "unset",
            exc,
        )
    _mark_perf("db_defaults_ms")
    if "request_context" not in locals():
        request_context = load_monitor_request_context(
            conn=None,
            resident_id=resident_id,
            input_source=input_source,
            mode=mode,
            dataset_code=dataset_code,
            op_code=op_code,
            event_location=payload_d.get("location"),
            active_model_code=active_model_code,
            requested_use_mc=requested_use_mc,
            requested_mc_M=requested_mc_M,
            is_replay=is_replay,
            ensure_system_settings_schema=lambda _conn: None,
            detect_variants=lambda _conn: {},
            table_exists=lambda _conn, _table: False,
        )

    event_location = request_context.event_location
    dataset_code = request_context.dataset_code
    op_code = request_context.op_code
    active_model_code = request_context.active_model_code
    cooldown_sec = request_context.cooldown_sec
    requested_use_mc = request_context.requested_use_mc
    requested_mc_M = request_context.requested_mc_M
    effective_use_mc = request_context.effective_use_mc
    effective_mc_M = request_context.effective_mc_M
    runtime = request_context.runtime

    expected_fps = {
        "le2i": 25,
        "caucafall": 23,
    }.get(dataset_code, int(payload_d.get("target_fps") or payload_d.get("fps") or 23))
    effective_fps = (
        float(expected_fps)
        if is_replay
        else _effective_target_fps(
            expected_fps=float(expected_fps),
            raw_fps_est=raw_stats.get("raw_fps_est"),
        )
    )

    if raw_xy is not None and raw_t_ms is not None:
        xy, conf, _, _, cap_fps_est = _resample_pose_window(
            raw_t_ms=raw_t_ms,
            raw_xy=raw_xy,
            raw_conf=raw_conf,
            target_fps=float(effective_fps),
            target_T=target_T,
            window_end_t_ms=float(window_end_t_ms) if window_end_t_ms is not None else None,
        )
        if xy:
            xy, conf = _preprocess_online_raw_window(xy, conf)

    if not xy:
        xy = payload_d.get("xy") or []
        conf = payload_d.get("conf") or []

    # Direct `xy/conf` payloads are already fixed-length windows; synthesize
    # timing/coverage stats so live quality gates don't treat them as empty raw streams.
    if (
        (raw_xy is None or raw_t_ms is None)
        and isinstance(xy, list)
        and len(xy) > 0
        and (raw_stats.get("raw_len") or 0) <= 0
    ):
        raw_stats = _direct_window_stats(xy, conf, effective_fps=float(effective_fps))

    if not xy:
        raise HTTPException(status_code=400, detail="payload must include raw_* (preferred) or xy")
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

    st = get_monitor_session_state(core._SESSION_STATE, session_id, _t_s)
    st_trackers = st["trackers"]
    st_trackers_cfg = st["trackers_cfg"]
    started_tcn = False
    started_gcn = False

    specs = _get_deploy_specs()

    def resolve_spec_key(arch: str, preferred: str) -> str:
        if preferred in specs:
            return preferred
        ds_prefix = f"{dataset_code}_"
        suffix = f"_{arch}"
        candidates = [k for k in specs.keys() if k.startswith(ds_prefix) and k.endswith(suffix)]
        if not candidates:
            return preferred
        candidates.sort(key=lambda x: (len(x), x))
        return candidates[0]

    def spec_key_for(arch: str) -> str:
        return f"{dataset_code}_{arch}".lower()

    tcn_key = resolve_spec_key("tcn", str(payload_d.get("model_tcn") or spec_key_for("tcn")).lower())
    gcn_key = resolve_spec_key("gcn", str(payload_d.get("model_gcn") or spec_key_for("gcn")).lower())
    has_tcn = tcn_key in specs
    has_gcn = gcn_key in specs
    if mode == "tcn":
        if not has_tcn:
            raise HTTPException(status_code=404, detail=f"No TCN deploy spec found for dataset '{dataset_code}'.")
    elif mode == "gcn":
        if not has_gcn:
            raise HTTPException(status_code=404, detail=f"No GCN deploy spec found for dataset '{dataset_code}'.")
    else:
        if not has_tcn and not has_gcn:
            raise HTTPException(
                status_code=404,
                detail=f"No deploy specs found for dataset '{dataset_code}'.",
            )
        if has_tcn and not has_gcn:
            mode = "tcn"
        elif has_gcn and not has_tcn:
            mode = "gcn"
        elif not has_tcn or not has_gcn:
            raise HTTPException(
                status_code=404,
                detail=f"Hybrid mode requires both TCN and GCN deploy specs for dataset '{dataset_code}'.",
            )

    guard_spec_key = tcn_key if mode in {"tcn", "hybrid"} else gcn_key
    primary_spec_key = tcn_key if mode == "tcn" else gcn_key if mode == "gcn" else tcn_key
    primary_model_key = "tcn" if mode == "tcn" else "gcn" if mode == "gcn" else "tcn"
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
    core.LAST_PRED_LATENCY_MS = latency_ms
    core.LAST_PRED_P_FALL = float(p_display)
    core.LAST_PRED_DECISION = str(triage_state)
    core.LAST_PRED_MODEL_CODE = str(active_model_code)
    core.LAST_PRED_TS_ISO = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
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
