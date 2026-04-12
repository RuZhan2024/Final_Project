from __future__ import annotations

from typing import Any, Dict, Optional

from .code_normalization import norm_op_code, normalize_dataset_code, normalize_model_code
from .db_schema import cols
from .deploy_runtime import get_specs as get_deploy_specs
from .services.monitor_uncertainty_service import resolve_uncertainty_cfg


def detect_variants(conn) -> Dict[str, str]:
    events_cols = cols(conn, "events")
    ops_cols = cols(conn, "operating_points")
    events_v = "v2" if "event_time" in events_cols else "v1"
    ops_v = "v2" if "model_id" in ops_cols else "v1"
    return {"settings": "v2", "events": events_v, "ops": ops_v}


def derive_ops_params_from_yaml(dataset_code: str, model_code: str, op_code: str) -> Dict[str, Any]:
    specs = get_deploy_specs()
    ds = normalize_dataset_code(dataset_code, default="caucafall")
    mc = normalize_model_code(model_code, default="TCN")
    oc = norm_op_code(op_code)

    def get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        try:
            if hasattr(obj, name):
                return getattr(obj, name)
        except (AttributeError, TypeError):
            pass
        if isinstance(obj, dict):
            return obj.get(name, default)
        return default

    def lookup_op_entry(ops: Dict[str, Any], normalized_code: str) -> Dict[str, Any]:
        candidates = [
            normalized_code,
            normalized_code.replace("-", ""),
            normalized_code.lower(),
            normalized_code.replace("-", "").lower(),
        ]
        for candidate in candidates:
            entry = ops.get(candidate)
            if isinstance(entry, dict):
                return dict(entry)

        for candidate in ["OP-2", "OP2", "op-2", "op2"]:
            entry = ops.get(candidate)
            if isinstance(entry, dict):
                return dict(entry)
        return {}

    def pack(spec_key: str) -> Optional[Dict[str, Any]]:
        spec = specs.get(spec_key)
        if spec is None:
            return None
        alert_cfg = dict(get_attr_or_key(spec, "alert_cfg", {}) or {})
        ops = dict(get_attr_or_key(spec, "ops", {}) or {})
        op = lookup_op_entry(ops, oc)
        return {"spec_key": spec_key, "alert_cfg": alert_cfg, "op": op}

    tcn = pack(f"{ds}_tcn")
    gcn = pack(f"{ds}_gcn")

    def tau(pack_: Optional[Dict[str, Any]], key: str, default: float) -> float:
        try:
            if pack_ and isinstance(pack_.get("op"), dict) and pack_["op"].get(key) is not None:
                return float(pack_["op"][key])
        except (TypeError, ValueError, KeyError):
            pass
        return float(default)

    def alert_cfg_value(pack_: Optional[Dict[str, Any]], key: str, default: float) -> float:
        try:
            if pack_ and isinstance(pack_.get("alert_cfg"), dict) and pack_["alert_cfg"].get(key) is not None:
                return float(pack_["alert_cfg"][key])
        except (TypeError, ValueError, KeyError):
            pass
        return float(default)

    def op_or_alert_cfg(pack_: Optional[Dict[str, Any]], key: str, default: float) -> float:
        try:
            if pack_ and isinstance(pack_.get("op"), dict) and pack_["op"].get(key) is not None:
                return float(pack_["op"][key])
        except (TypeError, ValueError, KeyError):
            pass
        return alert_cfg_value(pack_, key, default)

    if mc == "TCN":
        tau_low = tau(tcn, "tau_low", 0.0)
        tau_high = tau(tcn, "tau_high", 0.85)
        cooldown_s = op_or_alert_cfg(tcn, "cooldown_s", 3.0)
        ema_alpha = op_or_alert_cfg(tcn, "ema_alpha", 0.0)
        k = int(op_or_alert_cfg(tcn, "k", 2))
        n = int(op_or_alert_cfg(tcn, "n", 3))
        uncertainty_cfg = resolve_uncertainty_cfg(
            dict(tcn.get("alert_cfg") or {}) if tcn else {},
            dict(tcn.get("op") or {}) if tcn else {},
        )
    elif mc == "GCN":
        tau_low = tau(gcn, "tau_low", 0.0)
        tau_high = tau(gcn, "tau_high", 0.85)
        cooldown_s = op_or_alert_cfg(gcn, "cooldown_s", 3.0)
        ema_alpha = op_or_alert_cfg(gcn, "ema_alpha", 0.0)
        k = int(op_or_alert_cfg(gcn, "k", 2))
        n = int(op_or_alert_cfg(gcn, "n", 3))
        uncertainty_cfg = resolve_uncertainty_cfg(
            dict(gcn.get("alert_cfg") or {}) if gcn else {},
            dict(gcn.get("op") or {}) if gcn else {},
        )
    else:
        tau_low = tau(tcn, "tau_low", tau(gcn, "tau_low", 0.0))
        tau_high = tau(tcn, "tau_high", tau(gcn, "tau_high", 0.85))
        cooldown_s = op_or_alert_cfg(tcn, "cooldown_s", op_or_alert_cfg(gcn, "cooldown_s", 3.0))
        ema_alpha = op_or_alert_cfg(tcn, "ema_alpha", op_or_alert_cfg(gcn, "ema_alpha", 0.0))
        k = int(op_or_alert_cfg(tcn, "k", op_or_alert_cfg(gcn, "k", 2)))
        n = int(op_or_alert_cfg(tcn, "n", op_or_alert_cfg(gcn, "n", 3)))
        uncertainty_cfg = resolve_uncertainty_cfg(
            dict(tcn.get("alert_cfg") or gcn.get("alert_cfg") or {}) if (tcn or gcn) else {},
            dict(tcn.get("op") or gcn.get("op") or {}) if (tcn or gcn) else {},
        )

    return {
        "ui": {
            "op_code": oc,
            "tau_low": float(tau_low),
            "tau_high": float(tau_high),
            "cooldown_s": float(cooldown_s),
            "ema_alpha": float(ema_alpha),
            "k": int(k),
            "n": int(n),
            "mc_boundary_margin": float(uncertainty_cfg.get("boundary_margin", 0.08)),
            "mc_sigma_fall_max": float(uncertainty_cfg.get("sigma_fall_max", 0.08)),
        },
        "tcn": tcn if mc in {"TCN", "HYBRID"} else None,
        "gcn": gcn if mc in {"GCN", "HYBRID"} else None,
    }
