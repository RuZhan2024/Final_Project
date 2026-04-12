from __future__ import annotations

from typing import Any, Dict

from ..core import _norm_op_code, normalize_dataset_code, normalize_model_code
from ..inmemory_state import get_inmem_settings


def apply_yaml_override(system: Dict[str, Any], derive_ops_params_from_yaml) -> None:
    """Override UI thresholds/cooldown using YAML-derived deploy params (best-effort)."""
    try:
        system.setdefault("active_dataset_code", "caucafall")
        system.setdefault("active_model_code", "TCN")
        system.setdefault("active_op_code", "OP-2")
        dp = derive_ops_params_from_yaml(
            dataset_code=normalize_dataset_code(str(system.get("active_dataset_code") or "caucafall")),
            model_code=str(system.get("active_model_code") or "TCN"),
            op_code=str(system.get("active_op_code") or "OP-2"),
        )
        system["deploy_params"] = dp
        ui = dp.get("ui") or {}
        system["fall_threshold"] = float(ui.get("tau_high", system.get("fall_threshold", 0.71)))
        system["tau_low"] = float(ui.get("tau_low", system.get("tau_low", 0.0)))
        system["active_op_code"] = _norm_op_code(str(ui.get("op_code", system.get("active_op_code", "OP-2"))))
        system["alert_cooldown_sec"] = int(round(float(ui.get("cooldown_s", system.get("alert_cooldown_sec", 3)))))
    except (RuntimeError, OSError, TypeError, ValueError, KeyError):
        return


def build_settings_response(resident_id: int, system: Dict[str, Any], deploy: Dict[str, Any], *, db_available: bool) -> Dict[str, Any]:
    system["active_dataset_code"] = normalize_dataset_code(system.get("active_dataset_code"))
    system["active_model_code"] = normalize_model_code(system.get("active_model_code"))
    system["active_op_code"] = _norm_op_code(str(system.get("active_op_code") or "OP-2"))
    system["store_anonymized_data"] = bool(
        system.get("store_event_clips", False) and system.get("anonymize_skeleton_data", True)
    )

    return {
        "system": system,
        "deploy": deploy,
        "privacy": {
            "store_anonymized_data": system.get("store_anonymized_data", False),
            "store_event_clips": system.get("store_event_clips", False),
            "anonymize_skeleton_data": system.get("anonymize_skeleton_data", True),
        },
        "db_available": bool(db_available),
        "monitoring_enabled": system.get("monitoring_enabled", False),
        "active_model_code": system.get("active_model_code", "TCN"),
        "active_operating_point": system.get("active_operating_point"),
        "active_op_code": system.get("active_op_code"),
        "fall_threshold": system.get("fall_threshold", 0.71),
        "alert_cooldown_sec": system.get("alert_cooldown_sec", 3),
    }


def base_settings_snapshot(resident_id: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    base = get_inmem_settings(resident_id)
    return base["system"], base["deploy"]
