from __future__ import annotations

from typing import Any, Dict, Tuple


_DEFAULT_UNCERTAINTY_CFG = {
    "boundary_margin": 0.08,
    "sigma_fall_max": 0.08,
    "enabled": True,
}


def _coerce_float(value: Any, default: float) -> float:
    try:
        out = float(value)
        if out != out:
            return float(default)
        return float(out)
    except (TypeError, ValueError):
        return float(default)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def resolve_uncertainty_cfg(alert_cfg: Dict[str, Any], op_entry: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = dict(_DEFAULT_UNCERTAINTY_CFG)

    for source in (alert_cfg or {}, op_entry or {}):
        if not isinstance(source, dict):
            continue
        if isinstance(source.get("uncertainty_gate"), dict):
            gate = source["uncertainty_gate"]
            cfg["enabled"] = _coerce_bool(gate.get("enabled"), cfg["enabled"])
            cfg["boundary_margin"] = _coerce_float(gate.get("boundary_margin"), cfg["boundary_margin"])
            cfg["sigma_fall_max"] = _coerce_float(gate.get("sigma_fall_max"), cfg["sigma_fall_max"])
        if source.get("mc_boundary_margin") is not None:
            cfg["boundary_margin"] = _coerce_float(source.get("mc_boundary_margin"), cfg["boundary_margin"])
        if source.get("mc_sigma_fall_max") is not None:
            cfg["sigma_fall_max"] = _coerce_float(source.get("mc_sigma_fall_max"), cfg["sigma_fall_max"])
        if source.get("mc_uncertainty_enabled") is not None:
            cfg["enabled"] = _coerce_bool(source.get("mc_uncertainty_enabled"), cfg["enabled"])

    cfg["boundary_margin"] = max(0.01, min(0.30, float(cfg["boundary_margin"])))
    cfg["sigma_fall_max"] = max(0.005, min(0.30, float(cfg["sigma_fall_max"])))
    cfg["enabled"] = bool(cfg["enabled"])
    return cfg


def should_run_mc(
    *,
    use_mc: bool,
    p_det: float,
    tau_low: float,
    tau_high: float,
    uncertainty_cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    if not use_mc:
        return False, "disabled"
    if not bool(uncertainty_cfg.get("enabled", True)):
        return False, "gate_disabled"

    boundary_margin = float(uncertainty_cfg.get("boundary_margin", _DEFAULT_UNCERTAINTY_CFG["boundary_margin"]))
    lower = min(float(tau_low), float(tau_high)) - boundary_margin
    upper = max(float(tau_low), float(tau_high)) + boundary_margin
    if lower <= float(p_det) <= upper:
        return True, "boundary_window"
    return False, "outside_boundary"


def should_block_high_risk_fall(
    *,
    mc_applied: bool,
    sigma: float,
    probability: float,
    tau_high: float,
    uncertainty_cfg: Dict[str, Any],
) -> Tuple[bool, str | None]:
    if not mc_applied:
        return False, None
    if float(probability) < float(tau_high):
        return False, None
    sigma_fall_max = float(uncertainty_cfg.get("sigma_fall_max", _DEFAULT_UNCERTAINTY_CFG["sigma_fall_max"]))
    if float(sigma) > sigma_fall_max:
        return True, "high_uncertainty_fall_gate"
    return False, None


def apply_uncertainty_fall_gate(
    *,
    probability: float,
    sigma: float,
    tau_low: float,
    tau_high: float,
    mc_applied: bool,
    uncertainty_cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    blocked_fall, reason = should_block_high_risk_fall(
        mc_applied=mc_applied,
        sigma=sigma,
        probability=probability,
        tau_high=tau_high,
        uncertainty_cfg=uncertainty_cfg,
    )
    gated_probability = float(probability)
    if blocked_fall:
        gated_probability = float(min(gated_probability, float(tau_low) - 0.02))
    return gated_probability, {
        "mc_applied": bool(mc_applied),
        "sigma": float(sigma),
        "blocked_fall": bool(blocked_fall),
        "reason": reason,
    }
