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
    derive_ops_params_from_yaml as _derive_ops_params_from_yaml_impl,
    detect_variants as _detect_variants_impl,
)
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
from . import runtime_state as _runtime_state
from .schemas import CaregiverUpsertPayload, MonitorPredictPayload, SettingsUpdatePayload, SkeletonClipPayload

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


def _list_tables(conn) -> Set[str]:
    return _list_tables_impl(conn)


def _cols(conn, table: str) -> Set[str]:
    return _cols_impl(conn, table)


def _has_col(conn, table: str, col: str) -> bool:
    return _has_col_impl(conn, table, col)


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
    return _detect_variants_impl(conn)


# -----------------------------
# Deploy ops (YAML) helpers
# -----------------------------


def _norm_op_code(op_code: Optional[str]) -> str:
    return _norm_op_code_impl(op_code)


def _derive_ops_params_from_yaml(dataset_code: str, model_code: str, op_code: str) -> Dict[str, Any]:
    return _derive_ops_params_from_yaml_impl(dataset_code, model_code, op_code)


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
