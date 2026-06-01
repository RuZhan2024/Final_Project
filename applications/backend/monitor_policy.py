from __future__ import annotations

"""Operating-point policy helpers for the live monitor endpoint."""

from typing import Any, Dict

from fastapi import HTTPException

from .services.value_coercion import coerce_bool


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
    },
    "le2i": {
        "min_motion_for_fall": 0.020,
        "min_fps_ratio": 0.70,
        "min_conf_mean": 0.35,
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
    """Resolve live quality-gate thresholds for a deploy spec and op point.

    Dataset defaults provide the baseline, then YAML operating-point overrides
    can tune the guard. Numeric values are clamped before callers trust them.
    """

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
    except (TypeError, ValueError, AttributeError):
        pass

    # Treat deployment YAML as configuration, not trusted input.
    out["min_motion_for_fall"] = float(max(0.0, out["min_motion_for_fall"]))
    out["low_fps_mode_threshold"] = float(max(5.0, out["low_fps_mode_threshold"]))
    out["low_fps_fall_persist_n"] = int(max(1, out["low_fps_fall_persist_n"]))
    out["min_fps_ratio"] = float(min(1.5, max(0.1, out["min_fps_ratio"])))
    out["min_frames_ratio"] = float(min(1.0, max(0.1, out["min_frames_ratio"])))
    out["min_coverage_ratio"] = float(min(1.2, max(0.1, out["min_coverage_ratio"])))
    out["min_conf_mean"] = float(min(1.0, max(0.0, out["min_conf_mean"])))
    out["min_joints_med"] = int(max(1, out["min_joints_med"]))
    return out


def resolve_monitor_specs(
    *,
    specs: Dict[str, Any],
    dataset_code: str,
    mode: str,
    payload_d: Dict[str, Any],
) -> Dict[str, str]:
    """Select the single deploy spec used by the monitor runtime."""

    def resolve_spec_key(arch: str, preferred: str) -> str:
        """Use explicit spec keys when present, otherwise pick a dataset match."""

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

    if mode != "tcn":
        raise HTTPException(status_code=400, detail="Monitor runtime supports only mode='tcn'.")

    preferred = str(payload_d.get("model_tcn") or payload_d.get("model_id") or spec_key_for("tcn")).lower()
    tcn_key = resolve_spec_key("tcn", preferred)
    has_tcn = tcn_key in specs

    if not has_tcn:
        raise HTTPException(status_code=404, detail=f"No TCN deploy spec found for dataset '{dataset_code}'.")

    return {
        "mode": "tcn",
        "tcn_key": tcn_key,
        "guard_spec_key": tcn_key,
        "primary_spec_key": tcn_key,
    }
