# server/core.py
"""Shared helpers/state for the FastAPI backend.

This module centralises:
  - DB schema probing + lightweight caches
  - Pydantic payloads shared by multiple endpoints
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
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .deploy_runtime import get_specs as _get_deploy_specs

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass


# -----------------------------
# Payloads
# -----------------------------


class SettingsUpdatePayload(BaseModel):
    """Settings update payload.

    The UI sends a mostly-flat JSON body. We ignore unknown keys so the API
    remains backwards/forwards compatible.
    """

    model_config = ConfigDict(extra="ignore")

    monitoring_enabled: Optional[bool] = None
    api_online: Optional[bool] = None
    alert_cooldown_sec: Optional[int] = None
    notify_on_every_fall: Optional[bool] = None

    fall_threshold: Optional[float] = Field(
        default=None,
        description="Probability threshold for fall decision (usually 0.0–1.0).",
    )

    store_event_clips: Optional[bool] = None
    anonymize_skeleton_data: Optional[bool] = None
    store_anonymized_data: Optional[bool] = None

    active_model_code: Optional[str] = Field(default=None, description="TCN | GCN | HYBRID")
    active_operating_point: Optional[int] = Field(default=None, description="operating_points.id")
    active_dataset_code: Optional[str] = Field(default=None, description="le2i | caucafall")
    active_op_code: Optional[str] = Field(default=None, description="OP-1 | OP-2 | OP-3")

    mc_enabled: Optional[bool] = Field(default=None, description="Enable MC Dropout at inference")
    mc_M: Optional[int] = Field(default=None, description="MC samples for live inference")
    mc_M_confirm: Optional[int] = Field(default=None, description="MC samples for confirm step")

    # v2-style extras
    risk_profile: Optional[str] = None
    notify_sms: Optional[bool] = None
    notify_phone: Optional[bool] = None


class SkeletonClipPayload(BaseModel):
    """Skeleton-only event clip payload."""

    model_config = ConfigDict(extra="ignore")

    resident_id: int = 1
    dataset_code: Optional[str] = None
    mode: Optional[str] = None
    op_code: Optional[str] = None
    use_mc: Optional[bool] = None
    mc_M: Optional[int] = None
    mc_sigma_tol: Optional[float] = None
    mc_se_tol: Optional[float] = None
    pre_s: Optional[float] = None
    post_s: Optional[float] = None

    t_ms: List[float]
    xy: Optional[List[List[List[float]]]] = None
    conf: Optional[List[List[float]]] = None
    xy_flat: Optional[List[float]] = None
    conf_flat: Optional[List[float]] = None
    raw_joints: Optional[int] = None


class MonitorPredictPayload(BaseModel):
    """Live monitor inference payload.

    Keep fields permissive to preserve backward compatibility while making
    the request contract explicit and discoverable.
    """

    model_config = ConfigDict(extra="ignore")

    session_id: Optional[str] = None
    input_source: Optional[str] = None
    mode: Optional[str] = None
    dataset_code: Optional[str] = None
    dataset: Optional[str] = None
    op_code: Optional[str] = None
    op: Optional[str] = None

    model_tcn: Optional[str] = None
    model_gcn: Optional[str] = None
    model_id: Optional[str] = None

    resident_id: Optional[int] = None
    location: Optional[str] = None
    use_mc: Optional[bool] = None
    mc_M: Optional[int] = None
    persist: Optional[bool] = None
    compact_response: Optional[bool] = None

    target_T: Optional[int] = None
    target_fps: Optional[float] = None
    fps: Optional[float] = None
    capture_fps: Optional[float] = None
    timestamp_ms: Optional[float] = None
    window_end_t_ms: Optional[float] = None
    window_seq: Optional[int] = None

    raw_t_ms: Any = None
    raw_xy: Any = None
    raw_conf: Any = None
    raw_xy_q: Any = None
    raw_conf_q: Any = None
    raw_shape: Any = None
    xy: Any = None
    conf: Any = None


class CaregiverUpsertPayload(BaseModel):
    """Create or update a caregiver record."""

    model_config = ConfigDict(extra="ignore")

    id: Optional[int] = None
    resident_id: int = 1
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


# -----------------------------
# JSON helpers
# -----------------------------


def _jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, datetime):
        if x.tzinfo is None:
            return x.replace(tzinfo=timezone.utc).isoformat()
        return x.isoformat()
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except UnicodeDecodeError:
            return x.hex()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return str(x)


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
    """Return {'settings': 'v1|v2', 'events': 'v1|v2', 'ops': 'v1|v2'}."""
    settings_cols = _cols(conn, "system_settings")
    events_cols = _cols(conn, "events")
    ops_cols = _cols(conn, "operating_points")

    settings_v = "v2" if "active_model_id" in settings_cols else "v1"
    events_v = "v2" if "event_time" in events_cols else "v1"
    ops_v = "v2" if "model_id" in ops_cols else "v1"
    return {"settings": settings_v, "events": events_v, "ops": ops_v}


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

    def pack(spec_key: str) -> Optional[Dict[str, Any]]:
        s = specs.get(spec_key)
        if s is None:
            return None
        alert_cfg = dict(_get_attr_or_key(s, "alert_cfg", {}) or {})
        ops = dict(_get_attr_or_key(s, "ops", {}) or {})
        op = dict(ops.get(oc) or ops.get("OP-2") or {})
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
    elif mc == "GCN":
        tau_low = _tau(gcn, "tau_low", 0.0)
        tau_high = _tau(gcn, "tau_high", 0.85)
        cooldown_s = _op_or_acfg(gcn, "cooldown_s", 3.0)
        ema_alpha = _op_or_acfg(gcn, "ema_alpha", 0.0)
        k = int(_op_or_acfg(gcn, "k", 2))
        n = int(_op_or_acfg(gcn, "n", 3))
    else:
        # HYBRID UI defaults follow TCN-safe channel for primary auto-alert behavior.
        tau_low = _tau(tcn, "tau_low", _tau(gcn, "tau_low", 0.0))
        tau_high = _tau(tcn, "tau_high", _tau(gcn, "tau_high", 0.85))
        cooldown_s = _op_or_acfg(tcn, "cooldown_s", _op_or_acfg(gcn, "cooldown_s", 3.0))
        ema_alpha = _op_or_acfg(tcn, "ema_alpha", _op_or_acfg(gcn, "ema_alpha", 0.0))
        k = int(_op_or_acfg(tcn, "k", _op_or_acfg(gcn, "k", 2)))
        n = int(_op_or_acfg(tcn, "n", _op_or_acfg(gcn, "n", 3)))

    return {
        "ui": {
            "op_code": oc,
            "tau_low": float(tau_low),
            "tau_high": float(tau_high),
            "cooldown_s": float(cooldown_s),
            "ema_alpha": float(ema_alpha),
            "k": int(k),
            "n": int(n),
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


# -----------------------------
# Event clip persistence helpers
# -----------------------------


def _read_clip_privacy_flags(conn, resident_id: int) -> Tuple[bool, bool]:
    """Return (store_event_clips, anonymize_skeleton_data)."""
    store_event_clips = False
    anonymize = True
    try:
        _ensure_system_settings_schema(conn)
        variants = _detect_variants(conn)
        row = None
        with conn.cursor() as cur:
            if variants.get("settings") == "v2" and _table_exists(conn, "system_settings"):
                cur.execute(
                    "SELECT store_event_clips, anonymize_skeleton_data FROM system_settings WHERE resident_id=%s LIMIT 1",
                    (resident_id,),
                )
                row = cur.fetchone()
            elif _table_exists(conn, "settings"):
                cur.execute(
                    "SELECT store_event_clips, anonymize_skeleton_data FROM settings WHERE resident_id=%s LIMIT 1",
                    (resident_id,),
                )
                row = cur.fetchone()
        if isinstance(row, dict):
            if row.get("store_event_clips") is not None:
                store_event_clips = bool(int(row.get("store_event_clips"))) if str(row.get("store_event_clips")).isdigit() else bool(row.get("store_event_clips"))
            if row.get("anonymize_skeleton_data") is not None:
                anonymize = bool(int(row.get("anonymize_skeleton_data"))) if str(row.get("anonymize_skeleton_data")).isdigit() else bool(row.get("anonymize_skeleton_data"))
    except (MySQLError, RuntimeError, AttributeError, TypeError, ValueError):
        pass
    return store_event_clips, anonymize


def _event_clips_dir() -> Path:
    d = Path(__file__).resolve().parent / "event_clips"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def _anonymize_xy_inplace(xy: np.ndarray) -> np.ndarray:
    """Make coordinates relative to pelvis center (privacy-friendly)."""
    try:
        if xy.ndim != 3 or xy.shape[-1] != 2:
            return xy
        T, J, _ = xy.shape
        if J < 25:
            return xy
        pelvis = 0.5 * (xy[:, 23, :] + xy[:, 24, :])
        xy = xy - pelvis[:, None, :]
        return xy
    except (AttributeError, TypeError, ValueError, IndexError):
        return xy


# -----------------------------
# Caregivers schema helper
# -----------------------------


def _ensure_caregivers_table(conn) -> None:
    """Best-effort create caregivers table for dev environments."""
    if _table_exists(conn, "caregivers"):
        return
    try:
        with conn.cursor() as cur:
            backend = str(getattr(conn, "db_backend", "mysql")).lower()
            if backend == "sqlite":
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS caregivers (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      resident_id INTEGER NOT NULL,
                      name TEXT NULL,
                      email TEXT NULL,
                      phone TEXT NULL,
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
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                      INDEX idx_resident (resident_id)
                    )
                    """
                )
        conn.commit()
        global _TABLE_CACHE
        _TABLE_CACHE = None
    except (MySQLError, RuntimeError, AttributeError, TypeError, ValueError):
        pass


# -----------------------------
# In-memory settings fallback (when DB is not available)
# -----------------------------

_DEFAULT_SYSTEM_SETTINGS: Dict[str, Any] = {
    "monitoring_enabled": False,
    "api_online": True,
    "alert_cooldown_sec": 3,
    "notify_on_every_fall": True,
    "notify_sms": False,
    "notify_phone": False,
    "fall_threshold": 0.71,
    "store_event_clips": False,
    "anonymize_skeleton_data": True,
    "store_anonymized_data": False,
    "active_model_code": "TCN",
    "active_operating_point": None,
    "active_dataset_code": "caucafall",
    "active_op_code": "OP-2",
    "mc_enabled": False,
    "mc_M": 10,
    "mc_M_confirm": 25,
}

_DEFAULT_DEPLOY_SETTINGS: Dict[str, Any] = {
    "fps": 30,
    "window": {"W": 48, "S": 12},
    "mc": {"M": 10, "M_confirm": 25},
}

# resident_id -> {system, deploy}
_INMEM_SETTINGS: Dict[int, Dict[str, Any]] = {}
_INMEM_CAREGIVERS: Dict[int, List[Dict[str, Any]]] = {}


def get_inmem_settings(resident_id: int = 1) -> Dict[str, Any]:
    """Return a copy of the current in-memory settings."""
    rid = int(resident_id or 1)
    if rid not in _INMEM_SETTINGS:
        _INMEM_SETTINGS[rid] = {
            "system": dict(_DEFAULT_SYSTEM_SETTINGS),
            "deploy": json.loads(json.dumps(_DEFAULT_DEPLOY_SETTINGS)),
        }
    # deep-ish copy to avoid accidental mutation by callers
    return {
        "system": dict(_INMEM_SETTINGS[rid]["system"]),
        "deploy": json.loads(json.dumps(_INMEM_SETTINGS[rid]["deploy"])),
    }


def apply_settings_update_inmem(payload: SettingsUpdatePayload, resident_id: int = 1) -> None:
    """Apply a SettingsUpdatePayload to in-memory settings (best-effort)."""
    rid = int(resident_id or 1)
    cur = get_inmem_settings(rid)
    system = cur["system"]
    deploy = cur["deploy"]

    # Accept either 0-1 or 0-100.
    fall_thr = payload.fall_threshold
    if fall_thr is not None:
        try:
            v = float(fall_thr)
            if 1.0 < v <= 100.0:
                v = v / 100.0
            system["fall_threshold"] = max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            pass

    if payload.store_anonymized_data is not None:
        v = bool(payload.store_anonymized_data)
        system["store_event_clips"] = v
        system["anonymize_skeleton_data"] = v
        system["store_anonymized_data"] = v

    for k in [
        "monitoring_enabled",
        "api_online",
        "notify_on_every_fall",
        "notify_sms",
        "notify_phone",
        "store_event_clips",
        "anonymize_skeleton_data",
        "mc_enabled",
    ]:
        v = getattr(payload, k, None)
        if v is not None:
            system[k] = bool(v)

    if payload.notify_on_every_fall is False:
        system["notify_sms"] = False
        system["notify_phone"] = False

    system["store_anonymized_data"] = bool(
        system.get("store_event_clips", False) and system.get("anonymize_skeleton_data", True)
    )

    if payload.alert_cooldown_sec is not None:
        try:
            system["alert_cooldown_sec"] = int(payload.alert_cooldown_sec)
        except (TypeError, ValueError):
            pass

    if payload.active_model_code is not None:
        system["active_model_code"] = normalize_model_code(payload.active_model_code, default="TCN")

    if payload.active_operating_point is not None:
        system["active_operating_point"] = payload.active_operating_point

    if payload.active_dataset_code is not None:
        system["active_dataset_code"] = normalize_dataset_code(payload.active_dataset_code, default="caucafall")

    if payload.active_op_code is not None:
        system["active_op_code"] = str(payload.active_op_code).upper()

    if payload.mc_M is not None:
        try:
            system["mc_M"] = int(payload.mc_M)
            deploy.setdefault("mc", {})["M"] = int(payload.mc_M)
        except (TypeError, ValueError):
            pass

    if payload.mc_M_confirm is not None:
        try:
            system["mc_M_confirm"] = int(payload.mc_M_confirm)
            deploy.setdefault("mc", {})["M_confirm"] = int(payload.mc_M_confirm)
        except (TypeError, ValueError):
            pass

    _INMEM_SETTINGS[rid] = {"system": system, "deploy": deploy}


def get_inmem_caregivers(resident_id: int = 1) -> List[Dict[str, Any]]:
    """Return in-memory caregivers list for DB-offline fallback."""
    rid = int(resident_id or 1)
    rows = _INMEM_CAREGIVERS.get(rid, [])
    return [dict(r) for r in rows]


def upsert_inmem_caregiver(payload: CaregiverUpsertPayload) -> Dict[str, Any]:
    """Upsert caregiver in memory when DB is unavailable."""
    rid = int(payload.resident_id or 1)
    rows = _INMEM_CAREGIVERS.setdefault(rid, [])
    target_id = int(payload.id) if payload.id else (rows[0]["id"] if rows else 1)

    row = None
    for r in rows:
        if int(r.get("id", 0)) == target_id:
            row = r
            break

    now = datetime.now(timezone.utc).isoformat()
    if row is None:
        row = {
            "id": target_id,
            "resident_id": rid,
            "name": None,
            "email": None,
            "phone": None,
            "created_at": now,
            "updated_at": now,
        }
        rows.append(row)

    if payload.name is not None:
        row["name"] = payload.name
    if payload.email is not None:
        row["email"] = payload.email
    if payload.phone is not None:
        row["phone"] = payload.phone
    row["updated_at"] = now
    return dict(row)
