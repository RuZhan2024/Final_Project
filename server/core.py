# server/core.py
"""Shared helpers/state for the FastAPI backend.

This module centralises:
  - DB schema probing + lightweight caches
  - JSON-serialisation helpers
  - YAML-derived operating-point parameter helpers
  - In-memory monitor session state

Route handlers live under :mod:`server.routes.*`.
"""

from __future__ import annotations

import json
import time
import uuid

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException
import numpy as np

from . import db_schema as _db_schema_module
from .code_normalization import (
    norm_op_code as _norm_op_code_impl,
    normalize_dataset_code as _normalize_dataset_code_impl,
    normalize_model_code as _normalize_model_code_impl,
)
from .db_schema import (
    col_exists as _col_exists_impl,
    cols as _cols_impl,
    ensure_caregivers_table as _ensure_caregivers_table_impl,
    ensure_system_settings_schema as _ensure_system_settings_schema_impl,
    has_col as _has_col_impl,
    list_tables as _list_tables_impl,
    table_exists as _table_exists_impl,
)
from .deploy_ops import (
    detect_variants as _detect_variants_impl,
)
from .deploy_runtime import get_specs as _get_deploy_specs_impl
from .inmemory_state import (
    apply_settings_update_inmem as _apply_settings_update_inmem_impl,
    get_inmem_caregivers as _get_inmem_caregivers_impl,
    get_inmem_settings as _get_inmem_settings_impl,
    upsert_inmem_caregiver as _upsert_inmem_caregiver_impl,
)
from .event_schema import event_prob_col as _event_prob_col_impl, event_time_col as _event_time_col_impl
from .json_utils import jsonable as _jsonable_impl
from .repositories.models_repository import (
    resolve_model_code as _resolve_model_code_impl,
    resolve_model_id as _resolve_model_id_impl,
    resolve_op_id as _resolve_op_id_impl,
)
from .repositories.residents_repository import (
    one_resident_id as _one_resident_id_impl,
    resident_exists as _resident_exists_impl,
)
from .runtime_assets import (
    anonymize_xy_inplace as _anonymize_xy_inplace_impl,
    event_clips_dir as _event_clips_dir_impl,
    read_clip_privacy_flags as _read_clip_privacy_flags_impl,
)
from .services.monitor_uncertainty_service import resolve_uncertainty_cfg
from . import runtime_state as _runtime_state
from .schemas import CaregiverUpsertPayload, MonitorPredictPayload, SettingsUpdatePayload, SkeletonClipPayload


_TABLE_CACHE = _db_schema_module._TABLE_CACHE
_COL_CACHE = _db_schema_module._COL_CACHE


def _jsonable(x: Any) -> Any:
    return _jsonable_impl(x)


# -----------------------------
# Schema detection (cached)
# -----------------------------


def _list_tables(conn) -> Set[str]:
    global _TABLE_CACHE
    _db_schema_module._TABLE_CACHE = _TABLE_CACHE
    tables = _list_tables_impl(conn)
    _TABLE_CACHE = _db_schema_module._TABLE_CACHE
    return tables


def _cols(conn, table: str) -> Set[str]:
    if table in _COL_CACHE:
        return _COL_CACHE[table]
    try:
        with conn.cursor() as cur:
            backend = str(getattr(conn, "db_backend", "mysql")).lower()
            if backend == "sqlite":
                cur.execute(f"PRAGMA table_info(`{table}`)")
                rows = cur.fetchall() or []
            else:
                cur.execute(f"SHOW COLUMNS FROM `{table}`")
                rows = cur.fetchall() or []
    except Exception:
        _COL_CACHE[table] = set()
        return set()
    names = {
        row.get("Field") or row.get("name")
        for row in rows
        if isinstance(row, dict) and (row.get("Field") or row.get("name"))
    }
    _COL_CACHE[table] = set(names)
    return _COL_CACHE[table]


def _has_col(conn, table: str, col: str) -> bool:
    return col in _cols(conn, table)


def _table_exists(conn, table_name: str) -> bool:
    return _table_exists_impl(conn, table_name)


def _col_exists(conn, table_name: str, col_name: str) -> bool:
    return _col_exists_impl(conn, table_name, col_name)


def _ensure_system_settings_schema(conn) -> None:
    return _ensure_system_settings_schema_impl(conn)


def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def normalize_dataset_code(dataset_code: Optional[str], default: str = "caucafall") -> str:
    return _normalize_dataset_code_impl(dataset_code, default=default)


def normalize_model_code(model_code: Optional[str], default: str = "TCN") -> str:
    return _normalize_model_code_impl(model_code, default=default)


def _detect_variants(conn) -> Dict[str, str]:
    events_cols = _cols(conn, "events")
    ops_cols = _cols(conn, "operating_points")
    events_v = "v2" if "event_time" in events_cols else "v1"
    ops_v = "v2" if "model_id" in ops_cols else "v1"
    return {"settings": "v2", "events": events_v, "ops": ops_v}


def _get_deploy_specs():
    return _get_deploy_specs_impl()


# -----------------------------
# Deploy ops (YAML) helpers
# -----------------------------


def _norm_op_code(op_code: Optional[str]) -> str:
    return _norm_op_code_impl(op_code)


def _derive_ops_params_from_yaml(dataset_code: str, model_code: str, op_code: str) -> Dict[str, Any]:
    specs = _get_deploy_specs()
    ds = normalize_dataset_code(dataset_code, default="caucafall")
    mc = normalize_model_code(model_code, default="TCN")
    oc = _norm_op_code(op_code)

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


# -----------------------------
# In-memory session state (monitor demo)
# -----------------------------


_SESSION_STATE = _runtime_state.get_session_store()
SESSION_TTL_S = _runtime_state.SESSION_TTL_S
SESSION_MAX_STATES = _runtime_state.SESSION_MAX_STATES

LAST_PRED_LATENCY_MS = None
LAST_PRED_P_FALL = None
LAST_PRED_DECISION = None
LAST_PRED_MODEL_CODE = None
LAST_PRED_TS_ISO = None


def touch_session_state(session_id: str, now_s: Optional[float] = None) -> Dict[str, Any]:
    """Compatibility wrapper over :mod:`server.runtime_state` session state."""
    return _runtime_state.touch_session_state(session_id, now_s=now_s)


def prune_session_state(now_s: Optional[float] = None) -> int:
    """Compatibility wrapper over :mod:`server.runtime_state` pruning logic."""
    return _runtime_state.prune_session_state(
        now_s=now_s,
        ttl_s=SESSION_TTL_S,
        max_states=SESSION_MAX_STATES,
    )


# -----------------------------
# Generic DB helpers
# -----------------------------


def _one_resident_id(conn) -> int:
    return _one_resident_id_impl(conn)


def _resident_exists(conn, resident_id: int) -> bool:
    return _resident_exists_impl(conn, resident_id)


def _resolve_model_id(conn, model_code: str) -> Optional[int]:
    return _resolve_model_id_impl(conn, model_code)


def _resolve_model_code(conn, model_id: Optional[int]) -> Optional[str]:
    return _resolve_model_code_impl(conn, model_id)


def _resolve_op_id(conn, model_id: Optional[int], op_id: Optional[int]) -> Optional[int]:
    return _resolve_op_id_impl(conn, model_id, op_id)


def _event_time_col(conn) -> str:
    cols = _cols(conn, "events")
    for c in ("ts", "event_time", "created_at"):
        if c in cols:
            return c
    return "event_time"


def _event_prob_col(conn) -> Optional[str]:
    cols = _cols(conn, "events")
    for c in ("score", "p_fall"):
        if c in cols:
            return c
    return None


def _read_clip_privacy_flags(conn, resident_id: int) -> Tuple[bool, bool]:
    return _read_clip_privacy_flags_impl(
        conn,
        resident_id,
        ensure_system_settings_schema=_ensure_system_settings_schema,
        table_exists=_table_exists,
    )


def _event_clips_dir() -> Path:
    return _event_clips_dir_impl()


def _anonymize_xy_inplace(xy: np.ndarray) -> np.ndarray:
    return _anonymize_xy_inplace_impl(xy)


# -----------------------------
# Caregivers schema helper
# -----------------------------


def _ensure_caregivers_table(conn) -> None:
    return _ensure_caregivers_table_impl(conn)


def get_inmem_settings(resident_id: int = 1) -> Dict[str, Any]:
    return _get_inmem_settings_impl(resident_id)


def apply_settings_update_inmem(payload: SettingsUpdatePayload, resident_id: int = 1) -> None:
    _apply_settings_update_inmem_impl(
        payload,
        resident_id=resident_id,
        normalize_model_code=normalize_model_code,
        normalize_dataset_code=normalize_dataset_code,
        norm_op_code=_norm_op_code,
    )


def get_inmem_caregivers(resident_id: int = 1) -> List[Dict[str, Any]]:
    return _get_inmem_caregivers_impl(resident_id)


def upsert_inmem_caregiver(payload: CaregiverUpsertPayload) -> Dict[str, Any]:
    return _upsert_inmem_caregiver_impl(payload)
