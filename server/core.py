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
import os
import time
import uuid

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException

from .deploy_runtime import get_specs as _get_deploy_specs
from .inmemory_state import (
    apply_settings_update_inmem as _apply_settings_update_inmem_impl,
    get_inmem_caregivers as _get_inmem_caregivers_impl,
    get_inmem_settings as _get_inmem_settings_impl,
    upsert_inmem_caregiver as _upsert_inmem_caregiver_impl,
)
from .json_utils import jsonable as _jsonable_impl
from .runtime_assets import (
    anonymize_xy_inplace as _anonymize_xy_inplace_impl,
    event_clips_dir as _event_clips_dir_impl,
    read_clip_privacy_flags as _read_clip_privacy_flags_impl,
)
from .schemas import CaregiverUpsertPayload, MonitorPredictPayload, SettingsUpdatePayload, SkeletonClipPayload
from .services.monitor_uncertainty_service import resolve_uncertainty_cfg

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass


def _jsonable(x: Any) -> Any:
    return _jsonable_impl(x)


# -----------------------------
# Schema detection (cached)
# -----------------------------


_TABLE_CACHE: Optional[Set[str]] = None
_COL_CACHE: Dict[str, Set[str]] = {}


def _list_tables(conn) -> Set[str]:
    global _TABLE_CACHE
    if _TABLE_CACHE is not None:
        return _TABLE_CACHE
    try:
        with conn.cursor() as cur:
            backend = str(getattr(conn, "db_backend", "mysql")).lower()
            if backend == "sqlite":
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                rows = cur.fetchall() or []
            else:
                cur.execute("SHOW TABLES")
                rows = cur.fetchall() or []
        tables: Set[str] = set()
        for r in rows:
            if isinstance(r, dict):
                v = r.get("name")
                if v is None:
                    v = next(iter(r.values()), None)
                if v:
                    tables.add(str(v))
            else:
                try:
                    tables.add(str(r[0]))
                except (TypeError, IndexError, KeyError):
                    pass
        _TABLE_CACHE = tables
        return tables
    except (MySQLError, RuntimeError, AttributeError, TypeError, ValueError):
        _TABLE_CACHE = set()
        return set()


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
    except (MySQLError, RuntimeError, AttributeError, TypeError, ValueError):
        _COL_CACHE[table] = set()
        return set()
    cols = {
        r.get("Field") or r.get("name")
        for r in rows
        if isinstance(r, dict) and (r.get("Field") or r.get("name"))
    }
    _COL_CACHE[table] = set(cols)
    return _COL_CACHE[table]


def _has_col(conn, table: str, col: str) -> bool:
    return col in _cols(conn, table)


def _table_exists(conn, table_name: str) -> bool:
    return table_name in _list_tables(conn)


def _col_exists(conn, table_name: str, col_name: str) -> bool:
    return col_name in _cols(conn, table_name)


def _ensure_system_settings_schema(conn) -> None:
    """Best-effort dev-time migration for optional settings columns."""
    try:
        if not _table_exists(conn, "system_settings"):
            return

        wanted: Dict[str, str] = {
            "fall_threshold": "DECIMAL(6,4) NULL",
            "store_event_clips": "TINYINT(1) NOT NULL DEFAULT 0",
            "anonymize_skeleton_data": "TINYINT(1) NOT NULL DEFAULT 1",
            "active_dataset_code": "VARCHAR(16) NOT NULL DEFAULT 'caucafall'",
            "active_op_code": "VARCHAR(8) NOT NULL DEFAULT 'OP-2'",
            "mc_enabled": "TINYINT(1) NOT NULL DEFAULT 0",
            "mc_M": "INT NOT NULL DEFAULT 10",
            "mc_M_confirm": "INT NOT NULL DEFAULT 25",
            "notify_sms": "TINYINT(1) NOT NULL DEFAULT 0",
            "notify_phone": "TINYINT(1) NOT NULL DEFAULT 0",
            "fps": "INT NOT NULL DEFAULT 30",
            "window_size": "INT NOT NULL DEFAULT 48",
            "stride": "INT NOT NULL DEFAULT 12",
        }

        alters: List[str] = []
        for col, ddl in wanted.items():
            if not _col_exists(conn, "system_settings", col):
                alters.append(f"ADD COLUMN `{col}` {ddl}")

        if alters:
            with conn.cursor() as cur:
                cur.execute(f"ALTER TABLE `system_settings` {', '.join(alters)}")
            conn.commit()
            _COL_CACHE.pop("system_settings", None)
    except (MySQLError, RuntimeError, AttributeError, TypeError, ValueError):
        return


def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


SUPPORTED_DATASETS = {"caucafall", "le2i"}


def normalize_dataset_code(dataset_code: Optional[str], default: str = "caucafall") -> str:
    ds = str(dataset_code or "").lower().strip()
    if ds in SUPPORTED_DATASETS:
        return ds
    return default


SUPPORTED_MODEL_CODES = {"TCN", "GCN", "HYBRID"}


def normalize_model_code(model_code: Optional[str], default: str = "TCN") -> str:
    mc = str(model_code or "").upper().strip()
    if mc in SUPPORTED_MODEL_CODES:
        return mc
    return default


def _detect_variants(conn) -> Dict[str, str]:
    """Return active schema variants for system-backed settings, events, and ops."""
    events_cols = _cols(conn, "events")
    ops_cols = _cols(conn, "operating_points")

    events_v = "v2" if "event_time" in events_cols else "v1"
    ops_v = "v2" if "model_id" in ops_cols else "v1"
    return {"settings": "v2", "events": events_v, "ops": ops_v}


# -----------------------------
# Deploy ops (YAML) helpers
# -----------------------------


def _norm_op_code(op_code: Optional[str]) -> str:
    s = (op_code or "").strip().upper().replace("_", "-")
    if s in {"OP1", "OP-1", "HIGH", "OP-01"}:
        return "OP-1"
    if s in {"OP2", "OP-2", "BALANCED", "OP-02"}:
        return "OP-2"
    if s in {"OP3", "OP-3", "LOW", "OP-03"}:
        return "OP-3"
    return "OP-2"


def _derive_ops_params_from_yaml(dataset_code: str, model_code: str, op_code: str) -> Dict[str, Any]:
    """Return YAML-derived params for Settings/Monitor UI."""
    specs = _get_deploy_specs()
    ds = normalize_dataset_code(dataset_code, default="caucafall")
    mc = normalize_model_code(model_code, default="TCN")
    oc = _norm_op_code(op_code)

    def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
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

    def _lookup_op_entry(ops: Dict[str, Any], normalized_code: str) -> Dict[str, Any]:
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

        fallback_candidates = ["OP-2", "OP2", "op-2", "op2"]
        for candidate in fallback_candidates:
            entry = ops.get(candidate)
            if isinstance(entry, dict):
                return dict(entry)
        return {}

    def pack(spec_key: str) -> Optional[Dict[str, Any]]:
        s = specs.get(spec_key)
        if s is None:
            return None
        alert_cfg = dict(_get_attr_or_key(s, "alert_cfg", {}) or {})
        ops = dict(_get_attr_or_key(s, "ops", {}) or {})
        op = _lookup_op_entry(ops, oc)
        return {"spec_key": spec_key, "alert_cfg": alert_cfg, "op": op}

    tcn = pack(f"{ds}_tcn")
    gcn = pack(f"{ds}_gcn")

    def _tau(p: Optional[Dict[str, Any]], k: str, default: float) -> float:
        try:
            if p and isinstance(p.get("op"), dict) and p["op"].get(k) is not None:
                return float(p["op"][k])
        except (TypeError, ValueError, KeyError):
            pass
        return float(default)

    def _acfg(p: Optional[Dict[str, Any]], k: str, default: float) -> float:
        try:
            if p and isinstance(p.get("alert_cfg"), dict) and p["alert_cfg"].get(k) is not None:
                return float(p["alert_cfg"][k])
        except (TypeError, ValueError, KeyError):
            pass
        return float(default)

    def _op_or_acfg(p: Optional[Dict[str, Any]], k: str, default: float) -> float:
        try:
            if p and isinstance(p.get("op"), dict) and p["op"].get(k) is not None:
                return float(p["op"][k])
        except (TypeError, ValueError, KeyError):
            pass
        return _acfg(p, k, default)

    if mc == "TCN":
        tau_low = _tau(tcn, "tau_low", 0.0)
        tau_high = _tau(tcn, "tau_high", 0.85)
        cooldown_s = _op_or_acfg(tcn, "cooldown_s", 3.0)
        ema_alpha = _op_or_acfg(tcn, "ema_alpha", 0.0)
        k = int(_op_or_acfg(tcn, "k", 2))
        n = int(_op_or_acfg(tcn, "n", 3))
        uncertainty_cfg = resolve_uncertainty_cfg(
            dict(tcn.get("alert_cfg") or {}) if tcn else {},
            dict(tcn.get("op") or {}) if tcn else {},
        )
    elif mc == "GCN":
        tau_low = _tau(gcn, "tau_low", 0.0)
        tau_high = _tau(gcn, "tau_high", 0.85)
        cooldown_s = _op_or_acfg(gcn, "cooldown_s", 3.0)
        ema_alpha = _op_or_acfg(gcn, "ema_alpha", 0.0)
        k = int(_op_or_acfg(gcn, "k", 2))
        n = int(_op_or_acfg(gcn, "n", 3))
        uncertainty_cfg = resolve_uncertainty_cfg(
            dict(gcn.get("alert_cfg") or {}) if gcn else {},
            dict(gcn.get("op") or {}) if gcn else {},
        )
    else:
        # HYBRID UI defaults follow TCN-safe channel for primary auto-alert behavior.
        tau_low = _tau(tcn, "tau_low", _tau(gcn, "tau_low", 0.0))
        tau_high = _tau(tcn, "tau_high", _tau(gcn, "tau_high", 0.85))
        cooldown_s = _op_or_acfg(tcn, "cooldown_s", _op_or_acfg(gcn, "cooldown_s", 3.0))
        ema_alpha = _op_or_acfg(tcn, "ema_alpha", _op_or_acfg(gcn, "ema_alpha", 0.0))
        k = int(_op_or_acfg(tcn, "k", _op_or_acfg(gcn, "k", 2)))
        n = int(_op_or_acfg(tcn, "n", _op_or_acfg(gcn, "n", 3)))
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


_SESSION_STATE: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_S = int(os.getenv("SESSION_TTL_S", "1800"))
SESSION_MAX_STATES = int(os.getenv("SESSION_MAX_STATES", "1000"))

LAST_PRED_LATENCY_MS: Optional[float] = None
LAST_PRED_P_FALL: Optional[float] = None
LAST_PRED_DECISION: Optional[str] = None
LAST_PRED_MODEL_CODE: Optional[str] = None
LAST_PRED_TS_ISO: Optional[str] = None


def touch_session_state(session_id: str, now_s: Optional[float] = None) -> Dict[str, Any]:
    """Mark a session as active and return its state dict."""
    t_s = float(now_s if now_s is not None else time.time())
    st = _SESSION_STATE.setdefault(str(session_id), {})
    st["last_seen_s"] = t_s
    return st


def prune_session_state(now_s: Optional[float] = None) -> int:
    """Drop stale sessions and cap max in-memory session entries."""
    if not _SESSION_STATE:
        return 0

    t_s = float(now_s if now_s is not None else time.time())
    ttl_s = max(60, int(SESSION_TTL_S))
    max_states = max(10, int(SESSION_MAX_STATES))
    cutoff = t_s - float(ttl_s)
    removed = 0

    stale_ids = []
    for sid, st in _SESSION_STATE.items():
        try:
            last_seen = float((st or {}).get("last_seen_s", 0.0) or 0.0)
        except Exception:
            last_seen = 0.0
        if last_seen < cutoff:
            stale_ids.append(sid)

    for sid in stale_ids:
        if _SESSION_STATE.pop(sid, None) is not None:
            removed += 1

    if len(_SESSION_STATE) > max_states:
        ordered = sorted(
            _SESSION_STATE.items(),
            key=lambda kv: float((kv[1] or {}).get("last_seen_s", 0.0) or 0.0),
        )
        overflow = len(_SESSION_STATE) - max_states
        for sid, _ in ordered[:overflow]:
            if _SESSION_STATE.pop(sid, None) is not None:
                removed += 1

    return removed


# -----------------------------
# Generic DB helpers
# -----------------------------


def _one_resident_id(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM residents ORDER BY id ASC LIMIT 1")
        row = cur.fetchone()
        if not row:
            raise HTTPException(
                status_code=500,
                detail="No residents found. Did you run create_db.sql + seed data?",
            )
        return int(row["id"])


def _resident_exists(conn, resident_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 AS ok FROM residents WHERE id=%s LIMIT 1", (resident_id,))
        return cur.fetchone() is not None


def _resolve_model_id(conn, model_code: str) -> Optional[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM models WHERE code=%s LIMIT 1", (model_code,))
        row = cur.fetchone()
        if row:
            return int(row["id"])
        cur.execute("SELECT id FROM models WHERE UPPER(family)=%s LIMIT 1", (model_code.upper(),))
        row = cur.fetchone()
        return int(row["id"]) if row else None


def _resolve_model_code(conn, model_id: Optional[int]) -> Optional[str]:
    if model_id is None:
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT code, family FROM models WHERE id=%s LIMIT 1", (model_id,))
        row = cur.fetchone()
        if not row:
            return None
        return row.get("code") or row.get("family")


def _resolve_op_id(conn, model_id: Optional[int], op_id: Optional[int]) -> Optional[int]:
    if op_id is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, model_id FROM operating_points WHERE id=%s LIMIT 1", (op_id,))
            row = cur.fetchone()
            if not row:
                return None
            if model_id is not None and row.get("model_id") not in (None, model_id):
                return None
            return int(row["id"])
    except (MySQLError, RuntimeError, AttributeError, TypeError, ValueError):
        return op_id


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
    """Best-effort create caregivers table for dev environments."""
    try:
        with conn.cursor() as cur:
            backend = str(getattr(conn, "db_backend", "mysql")).lower()
            if not _table_exists(conn, "caregivers"):
                if backend == "sqlite":
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS caregivers (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          resident_id INTEGER NOT NULL,
                          name TEXT NULL,
                          email TEXT NULL,
                          phone TEXT NULL,
                          telegram_chat_id TEXT NULL,
                          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                          updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                else:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS caregivers (
                          id INT AUTO_INCREMENT PRIMARY KEY,
                          resident_id INT NOT NULL,
                          name VARCHAR(120) NULL,
                          email VARCHAR(200) NULL,
                          phone VARCHAR(80) NULL,
                          telegram_chat_id VARCHAR(120) NULL,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                          INDEX idx_resident (resident_id)
                        )
                        """
                    )
            elif not _col_exists(conn, "caregivers", "telegram_chat_id"):
                if backend == "sqlite":
                    cur.execute("ALTER TABLE caregivers ADD COLUMN telegram_chat_id TEXT NULL")
                else:
                    cur.execute("ALTER TABLE caregivers ADD COLUMN `telegram_chat_id` VARCHAR(120) NULL")
        conn.commit()
        global _TABLE_CACHE
        _TABLE_CACHE = None
        _COL_CACHE.pop("caregivers", None)
    except (MySQLError, RuntimeError, AttributeError, TypeError, ValueError):
        pass


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
