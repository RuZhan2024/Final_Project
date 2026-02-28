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
    require_confirmation: Optional[bool] = None

    active_model_code: Optional[str] = Field(default=None, description="TCN | GCN | HYBRID")
    active_operating_point: Optional[int] = Field(default=None, description="operating_points.id")
    active_dataset_code: Optional[str] = Field(default=None, description="le2i | urfd | caucafall | muvim")
    active_op_code: Optional[str] = Field(default=None, description="OP-1 | OP-2 | OP-3")

    mc_enabled: Optional[bool] = Field(default=None, description="Enable MC Dropout at inference")
    mc_M: Optional[int] = Field(default=None, description="MC samples for live inference")
    mc_M_confirm: Optional[int] = Field(default=None, description="MC samples for confirm step")

    # v2-style extras
    risk_profile: Optional[str] = None
    notify_email: Optional[str] = None
    notify_sms: Optional[bool] = None


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
        except Exception:
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
            cur.execute("SHOW TABLES")
            rows = cur.fetchall() or []
        tables: Set[str] = set()
        for r in rows:
            if isinstance(r, dict):
                v = next(iter(r.values()), None)
                if v:
                    tables.add(str(v))
            else:
                try:
                    tables.add(str(r[0]))
                except Exception:
                    pass
        _TABLE_CACHE = tables
        return tables
    except Exception:
        _TABLE_CACHE = set()
        return set()


def _cols(conn, table: str) -> Set[str]:
    if table in _COL_CACHE:
        return _COL_CACHE[table]
    try:
        with conn.cursor() as cur:
            cur.execute(f"SHOW COLUMNS FROM `{table}`")
            rows = cur.fetchall() or []
    except Exception:
        _COL_CACHE[table] = set()
        return set()
    cols = {r.get("Field") for r in rows if isinstance(r, dict) and r.get("Field")}
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
            "require_confirmation": "TINYINT(1) NOT NULL DEFAULT 0",
            "store_event_clips": "TINYINT(1) NOT NULL DEFAULT 0",
            "anonymize_skeleton_data": "TINYINT(1) NOT NULL DEFAULT 1",
            "active_dataset_code": "VARCHAR(16) NOT NULL DEFAULT 'muvim'",
            "active_op_code": "VARCHAR(8) NOT NULL DEFAULT 'OP-2'",
            "mc_enabled": "TINYINT(1) NOT NULL DEFAULT 1",
            "mc_M": "INT NOT NULL DEFAULT 10",
            "mc_M_confirm": "INT NOT NULL DEFAULT 25",
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
    except Exception:
        return


def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


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
    ds = (dataset_code or "muvim").lower().strip()
    mc = (model_code or "HYBRID").upper().strip()
    oc = _norm_op_code(op_code)

    def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        try:
            if hasattr(obj, name):
                return getattr(obj, name)
        except Exception:
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
        except Exception:
            pass
        return float(default)

    def _acfg(p: Optional[Dict[str, Any]], k: str, default: float) -> float:
        try:
            if p and isinstance(p.get("alert_cfg"), dict) and p["alert_cfg"].get(k) is not None:
                return float(p["alert_cfg"][k])
        except Exception:
            pass
        return float(default)

    if mc == "TCN":
        tau_low = _tau(tcn, "tau_low", 0.0)
        tau_high = _tau(tcn, "tau_high", 0.85)
        cooldown_s = _acfg(tcn, "cooldown_s", 3.0)
        ema_alpha = _acfg(tcn, "ema_alpha", 0.0)
        k = int(_acfg(tcn, "k", 2))
        n = int(_acfg(tcn, "n", 3))
    elif mc == "GCN":
        tau_low = _tau(gcn, "tau_low", 0.0)
        tau_high = _tau(gcn, "tau_high", 0.85)
        cooldown_s = _acfg(gcn, "cooldown_s", 3.0)
        ema_alpha = _acfg(gcn, "ema_alpha", 0.0)
        k = int(_acfg(gcn, "k", 2))
        n = int(_acfg(gcn, "n", 3))
    else:
        tau_low = min(_tau(tcn, "tau_low", 0.0), _tau(gcn, "tau_low", 0.0))
        tau_high = min(_tau(tcn, "tau_high", 0.85), _tau(gcn, "tau_high", 0.85))
        cooldown_s = max(_acfg(tcn, "cooldown_s", 3.0), _acfg(gcn, "cooldown_s", 3.0))
        ema_alpha = max(_acfg(tcn, "ema_alpha", 0.0), _acfg(gcn, "ema_alpha", 0.0))
        k = int(max(_acfg(tcn, "k", 2), _acfg(gcn, "k", 2)))
        n = int(max(_acfg(tcn, "n", 3), _acfg(gcn, "n", 3)))

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
    except Exception:
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
    except Exception:
        pass
    return store_event_clips, anonymize


def _event_clips_dir() -> Path:
    d = Path(__file__).resolve().parent / "event_clips"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
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
    except Exception:
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
    except Exception:
        pass


# -----------------------------
# In-memory settings fallback (when DB is not available)
# -----------------------------

_DEFAULT_SYSTEM_SETTINGS: Dict[str, Any] = {
    "monitoring_enabled": False,
    "api_online": True,
    "alert_cooldown_sec": 3,
    "notify_on_every_fall": True,
    "fall_threshold": 0.85,
    "store_event_clips": False,
    "anonymize_skeleton_data": True,
    "require_confirmation": False,
    "active_model_code": "HYBRID",
    "active_operating_point": None,
    "active_dataset_code": "muvim",
    "active_op_code": "OP-2",
    "mc_enabled": True,
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
        except Exception:
            pass

    for k in [
        "monitoring_enabled",
        "api_online",
        "notify_on_every_fall",
        "store_event_clips",
        "anonymize_skeleton_data",
        "require_confirmation",
        "mc_enabled",
    ]:
        v = getattr(payload, k, None)
        if v is not None:
            system[k] = bool(v)

    if payload.alert_cooldown_sec is not None:
        try:
            system["alert_cooldown_sec"] = int(payload.alert_cooldown_sec)
        except Exception:
            pass

    if payload.active_model_code is not None:
        system["active_model_code"] = str(payload.active_model_code).upper()

    if payload.active_operating_point is not None:
        system["active_operating_point"] = payload.active_operating_point

    if payload.active_dataset_code is not None:
        system["active_dataset_code"] = str(payload.active_dataset_code).lower()

    if payload.active_op_code is not None:
        system["active_op_code"] = str(payload.active_op_code).upper()

    if payload.mc_M is not None:
        try:
            system["mc_M"] = int(payload.mc_M)
            deploy.setdefault("mc", {})["M"] = int(payload.mc_M)
        except Exception:
            pass

    if payload.mc_M_confirm is not None:
        try:
            system["mc_M_confirm"] = int(payload.mc_M_confirm)
            deploy.setdefault("mc", {})["M_confirm"] = int(payload.mc_M_confirm)
        except Exception:
            pass

    _INMEM_SETTINGS[rid] = {"system": system, "deploy": deploy}
