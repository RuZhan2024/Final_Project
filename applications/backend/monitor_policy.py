from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from fastapi import HTTPException

from .services.value_coercion import coerce_bool


DUAL_POLICY_CFG_CACHE: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
DEFAULT_LIVE_GUARD_GLOBAL = {
    "low_fps_mode_threshold": 16.0,
    "low_fps_fall_persist_n": 3,
    "min_frames_ratio": 0.60,
    "min_coverage_ratio": 0.85,
    "min_joints_med": 20,
}
DEFAULT_LIVE_GUARD_BY_DATASET = {
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


def op_live_guard(
    specs: Dict[str, Any],
    spec_key: str,
    op_code: str,
    dataset_code: str,
    *,
    norm_op_code,
) -> Dict[str, Any]:
    ds_defaults = DEFAULT_LIVE_GUARD_BY_DATASET.get(dataset_code, {})
    out = {
        "min_motion_for_fall": float(ds_defaults.get("min_motion_for_fall", 0.020)),
        "low_fps_mode_threshold": float(DEFAULT_LIVE_GUARD_GLOBAL.get("low_fps_mode_threshold", 16.0)),
        "low_fps_fall_persist_n": int(DEFAULT_LIVE_GUARD_GLOBAL.get("low_fps_fall_persist_n", 3)),
        "min_fps_ratio": float(ds_defaults.get("min_fps_ratio", 0.70)),
        "min_frames_ratio": float(DEFAULT_LIVE_GUARD_GLOBAL.get("min_frames_ratio", 0.60)),
        "min_coverage_ratio": float(DEFAULT_LIVE_GUARD_GLOBAL.get("min_coverage_ratio", 0.85)),
        "min_conf_mean": float(ds_defaults.get("min_conf_mean", 0.35)),
        "min_joints_med": int(DEFAULT_LIVE_GUARD_GLOBAL.get("min_joints_med", 20)),
        "enable_stale_drop": True,
        "enable_low_motion_gate": True,
        "enable_occlusion_gate": True,
        "enable_structural_gate": True,
        "enable_low_fps_persist_gate": True,
        "allow_low_motion_high_conf_bypass": coerce_bool(ds_defaults.get("allow_low_motion_high_conf_bypass"), False),
        "low_motion_high_conf_k": int(ds_defaults.get("low_motion_high_conf_k", 0) or 0),
        "low_motion_high_conf_max_lying": ds_defaults.get("low_motion_high_conf_max_lying"),
    }
    try:
        spec = specs.get(spec_key)
        ops = spec.ops if spec is not None and hasattr(spec, "ops") else {}
        op = (ops or {}).get(norm_op_code(op_code)) or {}
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
            out["enable_stale_drop"] = coerce_bool(lg.get("enable_stale_drop"), out["enable_stale_drop"])
            out["enable_low_motion_gate"] = coerce_bool(lg.get("enable_low_motion_gate"), out["enable_low_motion_gate"])
            out["enable_occlusion_gate"] = coerce_bool(lg.get("enable_occlusion_gate"), out["enable_occlusion_gate"])
            out["enable_structural_gate"] = coerce_bool(lg.get("enable_structural_gate"), out["enable_structural_gate"])
            out["enable_low_fps_persist_gate"] = coerce_bool(
                lg.get("enable_low_fps_persist_gate"), out["enable_low_fps_persist_gate"]
            )
            out["allow_low_motion_high_conf_bypass"] = coerce_bool(
                lg.get("allow_low_motion_high_conf_bypass"), out["allow_low_motion_high_conf_bypass"]
            )
            out["low_motion_high_conf_k"] = int(lg.get("low_motion_high_conf_k", out["low_motion_high_conf_k"]))
            if lg.get("low_motion_high_conf_max_lying") is not None:
                out["low_motion_high_conf_max_lying"] = float(lg.get("low_motion_high_conf_max_lying"))
    except (TypeError, ValueError, AttributeError):
        pass

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


def op_delivery_gate(specs: Dict[str, Any], spec_key: str, op_code: str, *, norm_op_code) -> Dict[str, Any]:
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
        op = (ops or {}).get(norm_op_code(op_code)) or {}
        gate = op.get("delivery_gate") if isinstance(op, dict) else {}
        if isinstance(gate, dict):
            out["enabled"] = coerce_bool(gate.get("enabled"), False)
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


def op_uncertain_promote(specs: Dict[str, Any], spec_key: str, op_code: str, *, norm_op_code) -> Dict[str, Any]:
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
        op = (ops or {}).get(norm_op_code(op_code)) or {}
        cfg = op.get("uncertain_promote") if isinstance(op, dict) else {}
        if isinstance(cfg, dict):
            out["enabled"] = coerce_bool(cfg.get("enabled"), False)
            out["video_only"] = coerce_bool(cfg.get("video_only"), True)
            if cfg.get("min_p_alert") is not None:
                out["min_p_alert"] = float(cfg.get("min_p_alert"))
            if cfg.get("min_motion") is not None:
                out["min_motion"] = float(cfg.get("min_motion"))
            if cfg.get("max_lying") is not None:
                out["max_lying"] = float(cfg.get("max_lying"))
    except (TypeError, ValueError, AttributeError):
        pass
    return out


def load_dual_policy_cfg(dataset_code: str, policy_name: str, op_code: str, *, norm_op_code) -> Optional[Dict[str, Any]]:
    key = (f"{dataset_code}:{policy_name}", norm_op_code(op_code))
    if key in DUAL_POLICY_CFG_CACHE:
        return DUAL_POLICY_CFG_CACHE[key]

    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "ops" / "dual_policy" / f"tcn_{dataset_code}_dual_{policy_name}.yaml"
    if not path.exists():
        DUAL_POLICY_CFG_CACHE[key] = None
        return None

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        DUAL_POLICY_CFG_CACHE[key] = None
        return None

    if not isinstance(data, dict):
        DUAL_POLICY_CFG_CACHE[key] = None
        return None

    cfg: Dict[str, Any] = {}
    if isinstance(data.get("alert_cfg"), dict):
        cfg.update(data.get("alert_cfg") or {})
    elif isinstance(data.get("alert_base"), dict):
        cfg.update(data.get("alert_base") or {})

    ops = data.get("ops") if isinstance(data.get("ops"), dict) else {}
    op_entry = None
    want = norm_op_code(op_code)
    for key_name, value in (ops or {}).items():
        if not isinstance(value, dict):
            continue
        normalized = str(key_name).strip().upper().replace("_", "-")
        if normalized in {want, want.replace("-", "")}:
            op_entry = value
            break
    if op_entry is None and isinstance(ops, dict):
        op_entry = (ops.get("OP2") or ops.get("OP-2") or ops.get("op2") or ops.get("op-2"))

    if isinstance(op_entry, dict):
        if op_entry.get("tau_low") is not None:
            cfg["tau_low"] = float(op_entry.get("tau_low"))
        if op_entry.get("tau_high") is not None:
            cfg["tau_high"] = float(op_entry.get("tau_high"))

    cfg.setdefault("ema_alpha", 0.2)
    cfg.setdefault("k", 2)
    cfg.setdefault("n", 3)
    cfg.setdefault("cooldown_s", 30.0)
    cfg.setdefault("tau_low", 0.5)
    cfg.setdefault("tau_high", 0.85)

    DUAL_POLICY_CFG_CACHE[key] = cfg
    return cfg


def resolve_monitor_specs(
    *,
    specs: Dict[str, Any],
    dataset_code: str,
    mode: str,
    payload_d: Dict[str, Any],
) -> Dict[str, str]:
    def resolve_spec_key(arch: str, preferred: str) -> str:
        if preferred in specs:
            return preferred
        ds_prefix = f"{dataset_code}_"
        suffix = f"_{arch}"
        candidates = [key for key in specs.keys() if key.startswith(ds_prefix) and key.endswith(suffix)]
        if not candidates:
            return preferred
        candidates.sort(key=lambda key: (len(key), key))
        return candidates[0]

    def spec_key_for(arch: str) -> str:
        return f"{dataset_code}_{arch}".lower()

    tcn_key = resolve_spec_key("tcn", str(payload_d.get("model_tcn") or spec_key_for("tcn")).lower())
    gcn_key = resolve_spec_key("gcn", str(payload_d.get("model_gcn") or spec_key_for("gcn")).lower())
    has_tcn = tcn_key in specs
    has_gcn = gcn_key in specs

    resolved_mode = mode
    if resolved_mode == "tcn":
        if not has_tcn:
            raise HTTPException(status_code=404, detail=f"No TCN deploy spec found for dataset '{dataset_code}'.")
    elif resolved_mode == "gcn":
        if not has_gcn:
            raise HTTPException(status_code=404, detail=f"No GCN deploy spec found for dataset '{dataset_code}'.")
    else:
        if not has_tcn and not has_gcn:
            raise HTTPException(status_code=404, detail=f"No deploy specs found for dataset '{dataset_code}'.")
        if has_tcn and not has_gcn:
            resolved_mode = "tcn"
        elif has_gcn and not has_tcn:
            resolved_mode = "gcn"
        elif not has_tcn or not has_gcn:
            raise HTTPException(
                status_code=404,
                detail=f"Hybrid mode requires both TCN and GCN deploy specs for dataset '{dataset_code}'.",
            )

    primary_spec_key = tcn_key if resolved_mode == "tcn" else gcn_key if resolved_mode == "gcn" else tcn_key
    return {
        "mode": resolved_mode,
        "tcn_key": tcn_key,
        "gcn_key": gcn_key,
        "guard_spec_key": tcn_key if resolved_mode in {"tcn", "hybrid"} else gcn_key,
        "primary_spec_key": primary_spec_key,
        "primary_model_key": "tcn" if resolved_mode in {"tcn", "hybrid"} else "gcn",
    }
