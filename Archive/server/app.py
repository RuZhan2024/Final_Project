# server/app.py
from __future__ import annotations

import json
import time

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# Real model inference (best.pt) + per-dataset thresholds + MC dropout
from server.deploy_runtime import get_specs as _get_deploy_specs
from server.deploy_runtime import predict_spec as _predict_spec
from server.deploy_runtime import fuse_hybrid as _fuse_hybrid

# Support running as package (uvicorn server.app:app) or from cwd (uvicorn app:app)

from .db import get_conn



app = FastAPI(title="Elder Fall Monitor API", version="0.2")

# CORS for local dev
_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Payloads
# -----------------------------
class SettingsUpdatePayload(BaseModel):
    # Ignore unknown keys so older/newer frontends won't break the API.
    model_config = ConfigDict(extra="ignore")

    # Common / v1-style fields (some may not exist in DB; we store what we can)
    monitoring_enabled: Optional[bool] = None
    api_online: Optional[bool] = None
    alert_cooldown_sec: Optional[int] = None
    notify_on_every_fall: Optional[bool] = None

    # Detection thresholds (sent by Settings.js)
    fall_threshold: Optional[float] = Field(
        default=None,
        description="Probability threshold for fall decision (usually 0.0–1.0).",
    )

    # Privacy / UI toggles (sent by Settings.js)
    store_event_clips: Optional[bool] = None
    anonymize_skeleton_data: Optional[bool] = None
    require_confirmation: Optional[bool] = None

    # Model selection
    active_model_code: Optional[str] = Field(
        default=None,
        description="TCN | GCN | HYBRID (or any code present in DB)",
    )
    active_operating_point: Optional[int] = Field(default=None, description="operating_points.id")

    # Dataset/model-runtime selection (4 datasets) + uncertainty
    active_dataset_code: Optional[str] = Field(
        default=None,
        description="le2i | urfd | caucafall | muvim",
    )
    mc_enabled: Optional[bool] = Field(default=None, description="Enable MC Dropout at inference")
    mc_M: Optional[int] = Field(default=None, description="MC samples for live inference")
    mc_M_confirm: Optional[int] = Field(default=None, description="MC samples for confirm step (if used)")

    # v2-style notification/profile fields
    risk_profile: Optional[str] = None
    notify_email: Optional[str] = None
    notify_sms: Optional[bool] = None


# -----------------------------
# JSON helpers
# -----------------------------
def _jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, (datetime,)):
        # Always ISO format
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
# Schema detection
# -----------------------------
_COL_CACHE: Dict[str, Set[str]] = {}

# -----------------------------
# In-memory session state (monitor demo)
# -----------------------------
# The frontend uses a "session_id" to identify a running camera/pose stream.
# We keep lightweight per-session state here (e.g., cooldown/last alert).
_SESSION_STATE: Dict[str, Dict[str, Any]] = {}

# Last inference stats (for dashboard/UI)
LAST_PRED_LATENCY_MS: Optional[float] = None
LAST_PRED_P_FALL: Optional[float] = None
LAST_PRED_DECISION: Optional[str] = None
LAST_PRED_MODEL_CODE: Optional[str] = None
LAST_PRED_TS_ISO: Optional[str] = None

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

def _model_name_for_code(model_code: str) -> str:
    code = (model_code or "").upper().strip()
    if code == "TCN":
        return "TCN"
    if code == "GCN":
        return "GCN"
    if code == "HYBRID":
        return "TCN+GCN"
    return code or "Unknown"


def _table_exists(conn, table_name: str) -> bool:
    """Return True if table exists in current database."""
    with conn.cursor() as cur:
        cur.execute("SELECT DATABASE() AS db")
        row = cur.fetchone()
        db = row.get("db") if isinstance(row, dict) else (row[0] if row else None)
        if not db:
            return False
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema=%s AND table_name=%s LIMIT 1",
            (db, table_name),
        )
        return cur.fetchone() is not None


def _col_exists(conn, table_name: str, col_name: str) -> bool:
    """Return True if column exists in a table in current database."""
    with conn.cursor() as cur:
        cur.execute("SELECT DATABASE() AS db")
        row = cur.fetchone()
        db = row.get("db") if isinstance(row, dict) else (row[0] if row else None)
        if not db:
            return False
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema=%s AND table_name=%s AND column_name=%s LIMIT 1",
            (db, table_name, col_name),
        )
        return cur.fetchone() is not None


def _ensure_system_settings_schema(conn) -> None:
    """Best-effort dev-time migration for optional settings columns.

    The frontend expects some settings fields (e.g., fall_threshold, privacy toggles).
    Older DBs may not have these columns yet. We try to add them if possible.
    """
    try:
        if not _table_exists(conn, "system_settings"):
            return

        wanted: Dict[str, str] = {
            # Keep this nullable so legacy rows don't break.
            "fall_threshold": "DECIMAL(6,4) NULL",
            "require_confirmation": "TINYINT(1) NOT NULL DEFAULT 0",
            "store_event_clips": "TINYINT(1) NOT NULL DEFAULT 0",
            "anonymize_skeleton_data": "TINYINT(1) NOT NULL DEFAULT 1",
            # Dataset selection (4 dataset trained checkpoints)
            "active_dataset_code": "VARCHAR(16) NOT NULL DEFAULT 'muvim'",
            # MC Dropout controls
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
            # Invalidate cache
            _COL_CACHE.pop("system_settings", None)
    except Exception:
        # If the DB user doesn't have ALTER privileges, just skip.
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
# Generic helpers
# -----------------------------
def _one_resident_id(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM residents ORDER BY id ASC LIMIT 1")
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=500, detail="No residents found. Did you run create_db.sql + seed data?")
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
        # fallback: match by family (HYBRID/GCN/TCN)
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
    # if v2, ensure op belongs to model_id if provided
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
    return "event_time"  # best guess

def _event_prob_col(conn) -> Optional[str]:
    cols = _cols(conn, "events")
    for c in ("score", "p_fall"):
        if c in cols:
            return c
    return None


# -----------------------------
# Health
# -----------------------------
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ts": datetime.utcnow().isoformat()}


# -----------------------------
# Models
# -----------------------------
@app.get("/api/models/summary")
def models_summary() -> Dict[str, Any]:
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM models ORDER BY id ASC")
                rows = cur.fetchall() or []
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to query models: {e}")
    return {"models": _jsonable(rows)}


# -----------------------------
# Deploy specs (real checkpoints)
# -----------------------------
@app.get("/api/deploy/specs")
def deploy_specs() -> Dict[str, Any]:
    """Return available dataset-specific specs discovered from outputs/reports.

    This is **not** DB-backed; it reflects what is actually available in the repo
    (trained best.pt + JSON reports with fitted operating points).
    """
    specs = _get_deploy_specs()
    out = []
    datasets = set()
    for key, s in specs.items():
        datasets.add(s.dataset_code)
        out.append({
            "spec_key": key,
            "dataset_code": s.dataset_code,
            "arch": s.arch,
            "ckpt": str(s.ckpt),
            "ops": s.ops,
        })
    out.sort(key=lambda d: (d["dataset_code"], d["arch"]))
    return {"specs": out, "datasets": sorted(datasets)}


# -----------------------------
# Operating points
# -----------------------------
@app.get("/api/operating_points")
def operating_points(model_code: str = Query(..., description="TCN | GCN | HYBRID")):
    """Return operating point presets for a model.

    This endpoint supports two schemas:
      - ops v2: operating_points has model_id and columns thr_detect / thr_low_conf / thr_high_conf
      - ops v1: operating_points has model_code and columns threshold_low / threshold_high / cooldown_seconds

    Response includes BOTH naming styles for frontend compatibility.
    """
    model_code = (model_code or "").upper().strip()
    if model_code not in {"TCN", "GCN", "HYBRID"}:
        raise HTTPException(status_code=400, detail="model_code must be one of: TCN, GCN, HYBRID")

    with get_conn() as conn:
        _ensure_system_settings_schema(conn)
        variants = _detect_variants(conn)
        with conn.cursor() as cur:
            if variants["ops"] == "v2":
                # v2 schema: ops per model_id
                cur.execute("SELECT id FROM models WHERE code=%s", (model_code,))
                m = cur.fetchone()
                if not m:
                    raise HTTPException(status_code=404, detail=f"Unknown model_code: {model_code}")
                model_id = int(m["id"])

                # NOTE: some DBs may not have est_fa24h/est_recall; select safely.
                cur.execute(
                    """
                    SELECT id, name, code,
                           thr_detect, thr_low_conf, thr_high_conf,
                           est_fa24h, est_recall
                    FROM operating_points
                    WHERE model_id=%s
                    ORDER BY code
                    """,
                    (model_id,),
                )
                rows = cur.fetchall() or []
                ops = []
                for r in rows:
                    thr_detect = float(r.get("thr_detect")) if r.get("thr_detect") is not None else None
                    thr_low = float(r.get("thr_low_conf")) if r.get("thr_low_conf") is not None else None
                    thr_high = float(r.get("thr_high_conf")) if r.get("thr_high_conf") is not None else None

                    ops.append({
                        "id": int(r["id"]),
                        "name": r.get("name"),
                        "code": r.get("code"),
                        # canonical v2 names
                        "thr_detect": thr_detect,
                        "thr_low_conf": thr_low,
                        "thr_high_conf": thr_high,
                        # v1-compat names used by some frontend versions
                        "threshold_low": thr_low,
                        "threshold_high": thr_high,
                        "cooldown_seconds": 3,
                        # optional estimates
                        "est_fa24h": float(r["est_fa24h"]) if r.get("est_fa24h") is not None else None,
                        "est_recall": float(r["est_recall"]) if r.get("est_recall") is not None else None,
                    })
            else:
                # v1 schema
                cur.execute(
                    """
                    SELECT id, model_code, name, threshold_low, threshold_high, cooldown_seconds
                    FROM operating_points
                    WHERE model_code=%s
                    ORDER BY id
                    """,
                    (model_code,),
                )
                rows = cur.fetchall() or []
                ops = []
                for r in rows:
                    thr_low = float(r.get("threshold_low")) if r.get("threshold_low") is not None else None
                    thr_high = float(r.get("threshold_high")) if r.get("threshold_high") is not None else None
                    ops.append({
                        "id": int(r["id"]),
                        "name": r.get("name"),
                        "code": r.get("code") or None,
                        "thr_detect": None,
                        "thr_low_conf": thr_low,
                        "thr_high_conf": thr_high,
                        "threshold_low": thr_low,
                        "threshold_high": thr_high,
                        "cooldown_seconds": int(r.get("cooldown_seconds") or 3),
                        "est_fa24h": None,
                        "est_recall": None,
                    })

    return {"model_code": model_code, "operating_points": ops}


@app.get("/api/settings")
def get_settings(resident_id: int = Query(1, description="Resident id")):
    """Return UI settings (nested + legacy flat fields)."""
    with get_conn() as conn:
        _ensure_system_settings_schema(conn)
        variants = _detect_variants(conn)

        # Defaults (safe even if DB has minimal schema)
        system: Dict[str, Any] = {
            # Runtime monitoring (camera) cannot be reliably auto-started after a reload.
            # Default to off; the frontend can still show a separate "enabled" preference.
            "monitoring_enabled": False,
            "active_model_code": "HYBRID",
            "active_operating_point": None,
            "camera_source": "webcam",
            "fall_threshold": 0.85,
            "alert_cooldown_sec": 3,
            "store_event_clips": False,
            "anonymize_skeleton_data": True,
            "require_confirmation": False,
            "notify_on_every_fall": True,
            # New: dataset selection (default to your primary dataset)
            "active_dataset_code": "muvim",
            # New: MC dropout controls
            "mc_enabled": True,
        }
        deploy: Dict[str, Any] = {
            "fps": 30,
            "window": {"W": 48, "S": 12},
            "mc": {"M": 10, "M_confirm": 25},
        }

        if _table_exists(conn, "settings"):
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM settings WHERE resident_id=%s LIMIT 1", (resident_id,))
                row = cur.fetchone() or {}
            system.update({
                "monitoring_enabled": bool(_safe_get(row, "monitoring_enabled", 0)),
                "active_model_code": _safe_get(row, "active_model_code", "HYBRID"),
                "active_operating_point": _safe_get(row, "active_operating_point", None),
                "fall_threshold": float(_safe_get(row, "fall_threshold", 0.85) or 0.85),
                "alert_cooldown_sec": int(_safe_get(row, "alert_cooldown_sec", 3) or 3),
                "store_event_clips": bool(_safe_get(row, "store_event_clips", 0)),
                "anonymize_skeleton_data": bool(_safe_get(row, "anonymize_skeleton_data", 1)),
                "require_confirmation": bool(_safe_get(row, "require_confirmation", 0)),
                "notify_on_every_fall": bool(_safe_get(row, "notify_on_every_fall", 1)),
            })
            deploy.update({
                "fps": int(_safe_get(row, "fps", 30) or 30),
                "window": {
                    "W": int(_safe_get(row, "window_size", 48) or 48),
                    "S": int(_safe_get(row, "stride", 12) or 12),
                },
            })
        else:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM system_settings WHERE resident_id=%s ORDER BY id ASC LIMIT 1", (resident_id,))
                sys_row = cur.fetchone() or {}

                if isinstance(sys_row, dict) and "monitoring_enabled" in sys_row:
                    system["monitoring_enabled"] = bool(sys_row.get("monitoring_enabled", 0))

                if isinstance(sys_row, dict) and sys_row.get("camera_source"):
                    system["camera_source"] = sys_row.get("camera_source")

                # Prefer direct active_model_code if present (some schemas store code instead of id)

                if isinstance(sys_row, dict) and sys_row.get("active_model_code"):

                    system["active_model_code"] = sys_row.get("active_model_code") or system["active_model_code"]


                active_model_id = sys_row.get("active_model_id") if isinstance(sys_row, dict) else None
                if active_model_id and _table_exists(conn, "models"):
                    cur.execute("SELECT * FROM models WHERE id=%s LIMIT 1", (active_model_id,))
                    mrow = cur.fetchone() or {}
                    if isinstance(mrow, dict):
                        system["active_model_code"] = mrow.get("model_code") or mrow.get("code") or system["active_model_code"]

                if isinstance(sys_row, dict):
                    system["active_operating_point"] = sys_row.get("active_operating_point_id") or sys_row.get("active_operating_point")

                # If we have an active operating point, prefer its thresholds.
                # This gives you persistence for fall_threshold even if an older schema
                # doesn't store it directly in system_settings.
                try:
                    op_id = system.get("active_operating_point")
                    if op_id and _table_exists(conn, "operating_points") and _col_exists(conn, "operating_points", "thr_detect"):
                        cur.execute(
                            "SELECT thr_detect, thr_low_conf, thr_high_conf FROM operating_points WHERE id=%s LIMIT 1",
                            (int(op_id),),
                        )
                        op_row = cur.fetchone() or {}
                        if isinstance(op_row, dict):
                            if op_row.get("thr_detect") is not None:
                                system["fall_threshold"] = float(op_row.get("thr_detect"))
                            # Expose these too; the frontend may display them.
                            if op_row.get("thr_low_conf") is not None:
                                system["thr_low_conf"] = float(op_row.get("thr_low_conf"))
                            if op_row.get("thr_high_conf") is not None:
                                system["thr_high_conf"] = float(op_row.get("thr_high_conf"))
                except Exception:
                    pass

                # optional columns if present
                for col, setter in [
                    ("p_fall_threshold", lambda v: system.__setitem__("fall_threshold", float(v))),
                    ("fall_threshold", lambda v: system.__setitem__("fall_threshold", float(v))),
                    ("alert_cooldown_sec", lambda v: system.__setitem__("alert_cooldown_sec", int(v))),
                    ("store_event_clips", lambda v: system.__setitem__("store_event_clips", bool(v))),
                    ("anonymize_skeleton_data", lambda v: system.__setitem__("anonymize_skeleton_data", bool(v))),
                    ("require_confirmation", lambda v: system.__setitem__("require_confirmation", bool(v))),
                    ("notify_on_every_fall", lambda v: system.__setitem__("notify_on_every_fall", bool(v))),
                    ("fps", lambda v: deploy.__setitem__("fps", int(v))),
                    ("window_size", lambda v: deploy["window"].__setitem__("W", int(v))),
                    ("stride", lambda v: deploy["window"].__setitem__("S", int(v))),
                    ("mc_M", lambda v: deploy["mc"].__setitem__("M", int(v))),
                    ("mc_M_confirm", lambda v: deploy["mc"].__setitem__("M_confirm", int(v))),
                    ("active_dataset_code", lambda v: system.__setitem__("active_dataset_code", str(v).lower())),
                    ("mc_enabled", lambda v: system.__setitem__("mc_enabled", bool(int(v)) if str(v).isdigit() else bool(v))),
                ]:
                    if isinstance(sys_row, dict) and col in sys_row and sys_row.get(col) is not None:
                        try:
                            setter(sys_row[col])
                        except Exception:
                            pass

                # If we don't store a dedicated fall_threshold column in system_settings,
                # derive it from the active operating point's thr_detect.
                try:
                    op_id = system.get("active_operating_point")
                    has_ft = bool(sys_row.get("fall_threshold") is not None or sys_row.get("p_fall_threshold") is not None)
                    if (not has_ft) and op_id and _table_exists(conn, "operating_points"):
                        cur.execute(
                            "SELECT thr_detect, thr_low_conf, thr_high_conf FROM operating_points WHERE id=%s LIMIT 1",
                            (op_id,),
                        )
                        op = cur.fetchone() or {}
                        if isinstance(op, dict):
                            if op.get("thr_detect") is not None:
                                system["fall_threshold"] = float(op["thr_detect"])
                            # Optional extras (frontend can use if present)
                            if op.get("thr_low_conf") is not None:
                                system["thr_low_conf"] = float(op["thr_low_conf"])
                            if op.get("thr_high_conf") is not None:
                                system["thr_high_conf"] = float(op["thr_high_conf"])
                except Exception:
                    pass

        return {
            "system": system,
            "deploy": deploy,
            "privacy": {
                "store_event_clips": system.get("store_event_clips", False),
                "anonymize_skeleton_data": system.get("anonymize_skeleton_data", True),
            },
            # legacy flat keys
            "monitoring_enabled": system["monitoring_enabled"],
            "active_model_code": system["active_model_code"],
            "active_operating_point": system["active_operating_point"],
            "fall_threshold": system["fall_threshold"],
            "alert_cooldown_sec": system["alert_cooldown_sec"],
        }

@app.put("/api/settings")
def update_settings(payload: SettingsUpdatePayload, resident_id: Optional[int] = None):
    """Update settings.

    The React UI sends a *flat* JSON payload (e.g. {"monitoring_enabled": true}).
    We update only columns that exist in the connected DB, so different schemas
    won't crash your server.
    """
    rid = int(resident_id or 1)

    # Accept either 0-1 (preferred) or 0-100 (percent) from clients.
    if payload.fall_threshold is not None:
        try:
            v = float(payload.fall_threshold)
            if v > 1.0 and v <= 100.0:
                payload.fall_threshold = v / 100.0
        except Exception:
            pass

    with get_conn() as conn:
        variants = _detect_variants(conn)

        if _table_exists(conn, "settings"):
            sets = []
            vals = []

            if payload.monitoring_enabled is not None:
                sets.append("monitoring_enabled=%s")
                vals.append(1 if payload.monitoring_enabled else 0)

            if payload.fall_threshold is not None:
                sets.append("fall_threshold=%s")
                vals.append(payload.fall_threshold)

            if payload.alert_cooldown_sec is not None:
                sets.append("alert_cooldown_sec=%s")
                vals.append(payload.alert_cooldown_sec)

            if payload.store_event_clips is not None:
                sets.append("store_event_clips=%s")
                vals.append(1 if payload.store_event_clips else 0)

            if payload.anonymize_skeleton_data is not None:
                sets.append("anonymize_skeleton_data=%s")
                vals.append(1 if payload.anonymize_skeleton_data else 0)

            if payload.require_confirmation is not None:
                sets.append("require_confirmation=%s")
                vals.append(1 if payload.require_confirmation else 0)

            if payload.notify_on_every_fall is not None:
                sets.append("notify_on_every_fall=%s")
                vals.append(1 if payload.notify_on_every_fall else 0)

            if payload.active_model_code is not None:
                sets.append("active_model_code=%s")
                vals.append(payload.active_model_code)

            if payload.active_operating_point is not None:
                sets.append("active_operating_point=%s")
                vals.append(payload.active_operating_point)

            if sets:
                sql = "UPDATE settings SET " + ", ".join(sets) + ", updated_at=NOW() WHERE resident_id=%s"
                vals.append(rid)
                with conn.cursor() as cur:
                    cur.execute(sql, tuple(vals))
                conn.commit()

            return {"ok": True}

        # v2: system_settings (id=1) + models + operating_points
        sets = []
        vals = []

        def add(col: str, expr: str, value: Any):
            if _col_exists(conn, "system_settings", col):
                sets.append(expr)
                vals.append(value)

        # monitoring toggle (needed by /Dashboard)
        if payload.monitoring_enabled is not None:
            add("monitoring_enabled", "monitoring_enabled=%s", 1 if payload.monitoring_enabled else 0)

        # fall threshold / cooldown etc (optional columns)
        if payload.fall_threshold is not None:
            if _col_exists(conn, "system_settings", "p_fall_threshold"):
                sets.append("p_fall_threshold=%s")
                vals.append(payload.fall_threshold)
            else:
                add("fall_threshold", "fall_threshold=%s", payload.fall_threshold)

        if payload.alert_cooldown_sec is not None:
            add("alert_cooldown_sec", "alert_cooldown_sec=%s", payload.alert_cooldown_sec)

        if payload.store_event_clips is not None:
            add("store_event_clips", "store_event_clips=%s", 1 if payload.store_event_clips else 0)

        if payload.anonymize_skeleton_data is not None:
            add("anonymize_skeleton_data", "anonymize_skeleton_data=%s", 1 if payload.anonymize_skeleton_data else 0)

        if payload.require_confirmation is not None:
            add("require_confirmation", "require_confirmation=%s", 1 if payload.require_confirmation else 0)

        if payload.notify_on_every_fall is not None:
            add("notify_on_every_fall", "notify_on_every_fall=%s", 1 if payload.notify_on_every_fall else 0)

        # Dataset selection + MC Dropout controls
        if payload.active_dataset_code is not None:
            add("active_dataset_code", "active_dataset_code=%s", (payload.active_dataset_code or "").lower())

        if payload.mc_enabled is not None:
            add("mc_enabled", "mc_enabled=%s", 1 if payload.mc_enabled else 0)

        if payload.mc_M is not None:
            add("mc_M", "mc_M=%s", int(payload.mc_M))

        if payload.mc_M_confirm is not None:
            add("mc_M_confirm", "mc_M_confirm=%s", int(payload.mc_M_confirm))
        # Active model selection
        if payload.active_model_code is not None:
            # Store code directly if schema supports it
            if _col_exists(conn, "system_settings", "active_model_code"):
                add("active_model_code", "active_model_code=%s", payload.active_model_code)

            # If schema stores model as FK, resolve id
            if _col_exists(conn, "system_settings", "active_model_id") and _table_exists(conn, "models"):
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM models WHERE model_code=%s OR code=%s LIMIT 1",
                        (payload.active_model_code, payload.active_model_code),
                    )
                    mrow = cur.fetchone()
                if mrow and isinstance(mrow, dict) and mrow.get("id") is not None:
                    sets.append("active_model_id=%s")
                    vals.append(int(mrow["id"]))

        # Active operating point id
        if payload.active_operating_point is not None:
            if _col_exists(conn, "system_settings", "active_operating_point_id"):
                sets.append("active_operating_point_id=%s")
                vals.append(payload.active_operating_point)
            else:
                add("active_operating_point", "active_operating_point=%s", payload.active_operating_point)

        if sets:
            # Ensure a settings row exists for this resident (dev-friendly)
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM system_settings WHERE resident_id=%s ORDER BY id ASC LIMIT 1", (rid,))
                row = cur.fetchone()
                if not row:
                    cur.execute("INSERT INTO system_settings (resident_id) VALUES (%s)", (rid,))
                    conn.commit()

            sql = "UPDATE system_settings SET " + ", ".join(sets) + ", updated_at=NOW() WHERE resident_id=%s"
            vals.append(rid)
            with conn.cursor() as cur:
                cur.execute(sql, tuple(vals))
            conn.commit()

        # Keep operating point threshold in sync when the UI adjusts the main threshold.
        if payload.fall_threshold is not None and _table_exists(conn, "operating_points") and _col_exists(conn, "operating_points", "thr_detect"):
            try:
                op_id = payload.active_operating_point
                if op_id is None:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT active_operating_point_id, active_operating_point FROM system_settings WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
                            (rid,),
                        )
                        srow = cur.fetchone() or {}
                        if isinstance(srow, dict):
                            op_id = srow.get("active_operating_point_id") or srow.get("active_operating_point")
                if op_id is not None:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE operating_points SET thr_detect=%s, updated_at=NOW() WHERE id=%s",
                            (float(payload.fall_threshold), int(op_id)),
                        )
                    conn.commit()
            except Exception:
                pass

        return {"ok": True}

@app.get("/api/events")
def list_events(resident_id: Optional[int] = None, limit: int = 200) -> Dict[str, Any]:
    with get_conn() as conn:
        rid = resident_id if resident_id and _resident_exists(conn, resident_id) else _one_resident_id(conn)
        variants = _detect_variants(conn)
        time_col = _event_time_col(conn)
        prob_col = _event_prob_col(conn)

        with conn.cursor() as cur:
            if variants["events"] == "v2":
                # Join model code/family
                cols = _cols(conn, "events")
                select_prob = f"e.`{prob_col}` AS score," if prob_col else "NULL AS score,"
                cur.execute(
                    f"""SELECT e.id,
                                 e.`{time_col}` AS ts,
                                 e.`type` AS type,
                                 e.`status` AS status,
                                 {select_prob}
                                 e.operating_point_id,
                                 m.code AS model_code,
                                 m.family AS model_family,
                                 e.notes,
                                 e.fa24h_snapshot,
                                 e.payload_json
                          FROM events e
                          LEFT JOIN models m ON m.id = e.model_id
                          WHERE e.resident_id=%s
                          ORDER BY e.`{time_col}` DESC
                          LIMIT %s""",
                    (rid, limit),
                )
                rows = cur.fetchall() or []
                # Normalize meta
                out = []
                for r in rows:
                    meta = {}
                    if r.get("notes") is not None:
                        meta["notes"] = r.get("notes")
                    if r.get("fa24h_snapshot") is not None:
                        meta["fa24h_snapshot"] = r.get("fa24h_snapshot")
                    if r.get("payload_json") is not None:
                        meta["payload_json"] = r.get("payload_json")
                    out.append(
                        {
                            "id": r.get("id"),
                            "ts": r.get("ts"),
                            "type": r.get("type"),
                            "status": r.get("status"),
                            "score": r.get("score"),
                            "model_code": r.get("model_code") or r.get("model_family"),
                            "operating_point_id": r.get("operating_point_id"),
                            "meta": meta,
                        }
                    )
                return {"resident_id": rid, "events": _jsonable(out)}

            # v1 events schema
            cur.execute(
                f"""SELECT id, `{time_col}` AS ts, `type` AS type,
                             severity, model_code,
                             {('`'+prob_col+'` AS score,' if prob_col else 'NULL AS score,')}
                             meta
                      FROM events
                      WHERE resident_id=%s
                      ORDER BY `{time_col}` DESC
                      LIMIT %s""",
                (rid, limit),
            )
            rows = cur.fetchall() or []
            # Parse meta JSON if present
            out = []
            for r in rows:
                meta = r.get("meta")
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {"raw": meta}
                out.append(
                    {
                        "id": r.get("id"),
                        "ts": r.get("ts"),
                        "type": r.get("type"),
                        "severity": r.get("severity"),
                        "model_code": r.get("model_code"),
                        "score": r.get("score"),
                        "meta": meta,
                    }
                )
            return {"resident_id": rid, "events": _jsonable(out)}


@app.get("/api/events/summary")
def events_summary(resident_id: Optional[int] = None) -> Dict[str, Any]:
    with get_conn() as conn:
        rid = resident_id if resident_id and _resident_exists(conn, resident_id) else _one_resident_id(conn)
        time_col = _event_time_col(conn)

        since = datetime.utcnow() - timedelta(hours=24)
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT COUNT(*) AS n FROM events WHERE resident_id=%s", (rid,))
                total_events = int((cur.fetchone() or {}).get("n") or 0)

                cur.execute("SELECT COUNT(*) AS n FROM events WHERE resident_id=%s AND `type`='fall'", (rid,))
                total_falls = int((cur.fetchone() or {}).get("n") or 0)

                cur.execute(
                    f"SELECT COUNT(*) AS n FROM events WHERE resident_id=%s AND `{time_col}` >= %s",
                    (rid, since),
                )
                events_24h = int((cur.fetchone() or {}).get("n") or 0)

                cur.execute(
                    f"SELECT COUNT(*) AS n FROM events WHERE resident_id=%s AND `type`='fall' AND `{time_col}` >= %s",
                    (rid, since),
                )
                falls_24h = int((cur.fetchone() or {}).get("n") or 0)

                cur.execute(
                    f"SELECT * FROM events WHERE resident_id=%s ORDER BY `{time_col}` DESC LIMIT 1",
                    (rid,),
                )
                latest = cur.fetchone()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to compute events summary: {e}")

    return _jsonable(
        {
            "resident_id": rid,
            "total_events": total_events,
            "total_falls": total_falls,
            "events_last_24h": events_24h,
            "falls_last_24h": falls_24h,
            "latest_event": latest,
        }
    )


@app.post("/api/events/test_fall")
def test_fall() -> Dict[str, Any]:
    """Insert a synthetic 'fall' event for UI testing."""
    with get_conn() as conn:
        rid = _one_resident_id(conn)
        variants = _detect_variants(conn)
        now = datetime.utcnow()

        with conn.cursor() as cur:
            if variants["events"] == "v2":
                # pick active model/op from system_settings if present
                model_id = None
                op_id = None
                if _has_col(conn, "system_settings", "active_model_id"):
                    cur.execute("SELECT active_model_id, active_operating_point_id FROM system_settings WHERE resident_id=%s LIMIT 1", (rid,))
                    s = cur.fetchone() or {}
                    model_id = s.get("active_model_id")
                    op_id = s.get("active_operating_point_id")
                # Insert with best-effort columns
                cols = _cols(conn, "events")
                insert_cols = ["resident_id"]
                insert_vals = ["%s"]
                params: List[Any] = [rid]

                def add(col: str, val: Any):
                    if col in cols:
                        insert_cols.append(col)
                        insert_vals.append("%s")
                        params.append(val)

                add("model_id", model_id)
                add("operating_point_id", op_id)
                add("event_time", now)
                add("type", "fall")
                if "status" in cols:
                    add("status", "unreviewed")
                if "p_fall" in cols:
                    add("p_fall", 0.99)
                if "p_uncertain" in cols:
                    add("p_uncertain", 0.01)
                if "p_nonfall" in cols:
                    add("p_nonfall", 0.00)
                if "alert_sent" in cols:
                    add("alert_sent", 0)
                if "notes" in cols:
                    add("notes", "Test fall event (UI)")
                if "payload_json" in cols:
                    add("payload_json", json.dumps({"source": "ui_test"}))

                sql = f"INSERT INTO events ({', '.join('`'+c+'`' for c in insert_cols)}) VALUES ({', '.join(insert_vals)})"
                cur.execute(sql, tuple(params))
                new_id = cur.lastrowid
                cur.execute("SELECT * FROM events WHERE id=%s", (new_id,))
                row = cur.fetchone()
                return {"ok": True, "event": _jsonable(row)}

            # v1 insert
            meta = {"source": "ui_test"}
            cur.execute(
                """INSERT INTO events (resident_id, ts, type, severity, model_code, score, meta)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (rid, now, "fall", "high", "HYBRID", 0.99, json.dumps(meta)),
            )
            new_id = cur.lastrowid
            cur.execute("SELECT * FROM events WHERE id=%s", (new_id,))
            row = cur.fetchone()
            return {"ok": True, "event": _jsonable(row)}


# -----------------------------
# Dashboard summary
# -----------------------------
@app.get("/api/dashboard/summary")
def dashboard_summary():
    """Summary used by /Dashboard. Never returns 500."""
    global LAST_PRED_LATENCY_MS

    summary = {
        "status": "normal",
        "today": {"falls_detected": 0, "false_alarms": 0},
        "system": {
            "model_name": "HYBRID",
            "monitoring_enabled": False,
            "last_latency_ms": int(LAST_PRED_LATENCY_MS or 0),
            "api_online": True,
        },
    }

    try:
        with get_conn() as conn:
            # settings / model
            if _table_exists(conn, "system_settings"):
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM system_settings WHERE resident_id=%s ORDER BY id ASC LIMIT 1", (resident_id,))
                    sys_row = cur.fetchone() or {}
                    if isinstance(sys_row, dict) and "monitoring_enabled" in sys_row:
                        summary["system"]["monitoring_enabled"] = bool(sys_row.get("monitoring_enabled", 1))
                    # Prefer direct active_model_code if present (some schemas store code instead of id)
                    if isinstance(sys_row, dict) and sys_row.get("active_model_code"):
                        system["active_model_code"] = sys_row.get("active_model_code") or system["active_model_code"]

                    active_model_id = sys_row.get("active_model_id") if isinstance(sys_row, dict) else None
                    if active_model_id and _table_exists(conn, "models"):
                        cur.execute("SELECT * FROM models WHERE id=%s LIMIT 1", (active_model_id,))
                        mrow = cur.fetchone() or {}
                        if isinstance(mrow, dict):
                            summary["system"]["model_name"] = mrow.get("name") or mrow.get("model_code") or mrow.get("code") or summary["system"]["model_name"]
            elif _table_exists(conn, "settings") and _col_exists(conn, "settings", "monitoring_enabled"):
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM settings WHERE resident_id=%s LIMIT 1", (1,))
                    row = cur.fetchone() or {}
                    if isinstance(row, dict):
                        summary["system"]["monitoring_enabled"] = bool(row.get("monitoring_enabled", 1))
                        summary["system"]["model_name"] = row.get("active_model_code") or summary["system"]["model_name"]

            # counts today
            today_falls = 0
            today_false = 0
            with conn.cursor() as cur:
                if _table_exists(conn, "events") and _col_exists(conn, "events", "event_type"):
                    cur.execute(
                        "SELECT COUNT(*) AS c FROM events "
                        "WHERE DATE(created_at)=CURDATE() AND UPPER(event_type) IN ('FALL','FALL_DETECTED','FALL_CONFIRMED')"
                    )
                    r = cur.fetchone() or {}
                    today_falls = int(r.get("c", 0)) if isinstance(r, dict) else int(list(r)[0])
                    cur.execute(
                        "SELECT COUNT(*) AS c FROM events "
                        "WHERE DATE(created_at)=CURDATE() AND UPPER(event_type) IN ('FALSE_ALARM','FALSE','FALSE_POSITIVE')"
                    )
                    r = cur.fetchone() or {}
                    today_false = int(r.get("c", 0)) if isinstance(r, dict) else int(list(r)[0])
                elif _table_exists(conn, "fall_events"):
                    cur.execute(
                        "SELECT "
                        "SUM(CASE WHEN event_type='fall_detected' THEN 1 ELSE 0 END) AS falls_detected, "
                        "SUM(CASE WHEN event_type='false_alarm' THEN 1 ELSE 0 END) AS false_alarms "
                        "FROM fall_events WHERE DATE(created_at)=CURDATE()"
                    )
                    r = cur.fetchone() or {}
                    if isinstance(r, dict):
                        today_falls = int(r.get("falls_detected") or 0)
                        today_false = int(r.get("false_alarms") or 0)

            summary["today"]["falls_detected"] = today_falls
            summary["today"]["false_alarms"] = today_false
            summary["status"] = "alert" if today_falls > 0 else "normal"

            # heartbeat latency if available
            with conn.cursor() as cur:
                if _table_exists(conn, "heartbeat") and _col_exists(conn, "heartbeat", "latency_ms"):
                    cur.execute("SELECT latency_ms FROM heartbeat ORDER BY created_at DESC LIMIT 1")
                    r = cur.fetchone()
                    if isinstance(r, dict) and r.get("latency_ms") is not None:
                        summary["system"]["last_latency_ms"] = int(r["latency_ms"])

            return summary

    except Exception as e:
        summary["system"]["api_online"] = False
        summary["system"]["error"] = str(e)
        return summary


def _as_float_list(xs: Any) -> List[float]:
    if not isinstance(xs, list):
        return []
    out: List[float] = []
    for v in xs:
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def _resample_pose_window(
    *,
    raw_t_ms: Any,
    raw_xy: Any,
    raw_conf: Any = None,
    target_fps: float = 30.0,
    target_T: int = 48,
    window_end_t_ms: Optional[float] = None,
) -> Tuple[List[Any], List[Any], float, float, Optional[float]]:
    """Resample variable-FPS pose frames to a fixed FPS + fixed length window.

    Inputs (raw_*):
      - raw_t_ms: list of timestamps in milliseconds (monotonic, from performance.now() is OK)
      - raw_xy:  list length N; each frame is [J,2]
      - raw_conf:list length N; each frame is [J] (optional)
    Output:
      - xy_out:  list length target_T; each is [J,2]
      - conf_out:list length target_T; each is [J]
      - start_t_ms, end_t_ms: window alignment
      - capture_fps estimate (if timestamps valid)
    """
    # Validate
    if not isinstance(raw_t_ms, list) or not isinstance(raw_xy, list) or len(raw_t_ms) != len(raw_xy) or len(raw_xy) < 1:
        return [], [], 0.0, 0.0, None

    # Conf optional
    if isinstance(raw_conf, list) and len(raw_conf) == len(raw_xy):
        use_conf = True
    else:
        use_conf = False
        raw_conf = [None] * len(raw_xy)

    # Coerce timestamps + enforce strict monotonic increasing
    t: List[float] = []
    xy: List[Any] = []
    conf: List[Any] = []
    last_t: Optional[float] = None
    for ti, xyi, ci in zip(raw_t_ms, raw_xy, raw_conf):
        try:
            tf = float(ti)
        except Exception:
            continue
        if last_t is not None and tf <= last_t:
            # drop non-increasing samples (common when clocks jitter)
            continue
        if not isinstance(xyi, list):
            continue
        t.append(tf)
        xy.append(xyi)
        conf.append(ci if use_conf else None)
        last_t = tf

    if len(t) < 1:
        return [], [], 0.0, 0.0, None

    # Capture FPS estimate (based on raw timestamps)
    cap_fps: Optional[float] = None
    if len(t) >= 2:
        dt_s = (t[-1] - t[0]) / 1000.0
        if dt_s > 1e-6:
            cap_fps = (len(t) - 1) / dt_s

    # Target timeline
    target_fps = float(target_fps) if target_fps and float(target_fps) > 0 else 30.0
    target_T = int(target_T) if target_T and int(target_T) > 1 else 48
    dt_ms = 1000.0 / target_fps

    end_t_ms = float(window_end_t_ms) if window_end_t_ms is not None else t[-1]
    start_t_ms = end_t_ms - (target_T - 1) * dt_ms

    # Pointer for bracket search
    j = 0
    xy_out: List[Any] = []
    conf_out: List[Any] = []

    # Helper: linear interpolation on nested lists ([J,2] or [J])
    def lerp(a: float, b: float, alpha: float) -> float:
        return a + (b - a) * alpha

    def interp_frame(frame0: Any, frame1: Any, alpha: float, is_xy: bool) -> Any:
        # frame0 and frame1 are lists; keep shape and fallback safely
        if not isinstance(frame0, list):
            return frame1
        if not isinstance(frame1, list):
            return frame0
        out = []
        if is_xy:
            # [J,2]
            n = min(len(frame0), len(frame1))
            for k in range(n):
                p0 = frame0[k] if isinstance(frame0[k], list) and len(frame0[k]) >= 2 else [0.0, 0.0]
                p1 = frame1[k] if isinstance(frame1[k], list) and len(frame1[k]) >= 2 else [0.0, 0.0]
                try:
                    x0, y0 = float(p0[0]), float(p0[1])
                    x1, y1 = float(p1[0]), float(p1[1])
                except Exception:
                    x0 = y0 = x1 = y1 = 0.0
                out.append([lerp(x0, x1, alpha), lerp(y0, y1, alpha)])
            # If lengths differ, append remaining from the longer frame
            if len(frame0) > n:
                out.extend(frame0[n:])
            elif len(frame1) > n:
                out.extend(frame1[n:])
            return out
        else:
            # [J]
            n = min(len(frame0), len(frame1))
            for k in range(n):
                try:
                    v0 = float(frame0[k])
                except Exception:
                    v0 = 0.0
                try:
                    v1 = float(frame1[k])
                except Exception:
                    v1 = v0
                out.append(lerp(v0, v1, alpha))
            if len(frame0) > n:
                out.extend(frame0[n:])
            elif len(frame1) > n:
                out.extend(frame1[n:])
            return out

    for k in range(target_T):
        tw = start_t_ms + k * dt_ms

        if tw <= t[0]:
            xy_out.append(xy[0])
            conf_out.append(conf[0] if use_conf and isinstance(conf[0], list) else [1.0] * (len(xy[0]) if isinstance(xy[0], list) else 0))
            continue
        if tw >= t[-1]:
            xy_out.append(xy[-1])
            conf_out.append(conf[-1] if use_conf and isinstance(conf[-1], list) else [1.0] * (len(xy[-1]) if isinstance(xy[-1], list) else 0))
            continue

        while j + 1 < len(t) and t[j + 1] < tw:
            j += 1

        t0, t1 = t[j], t[j + 1]
        alpha = 0.0 if t1 <= t0 else (tw - t0) / (t1 - t0)

        xy_out.append(interp_frame(xy[j], xy[j + 1], alpha, True))

        if use_conf and isinstance(conf[j], list) and isinstance(conf[j + 1], list):
            conf_out.append(interp_frame(conf[j], conf[j + 1], alpha, False))
        else:
            # If no conf provided, keep all 1s.
            conf_out.append([1.0] * (len(xy_out[-1]) if isinstance(xy_out[-1], list) else 0))

    return xy_out, conf_out, start_t_ms, end_t_ms, cap_fps


@app.post("/api/monitor/reset_session")
def reset_session(session_id: str = Query(...)) -> Dict[str, Any]:
    """Reset transient in-memory state for a monitor session.

    The React /monitor-demo client may call this when restarting the camera
    pipeline or switching models. The backend doesn't need to know everything
    about the session, but providing this endpoint prevents 404s and lets us
    clear any server-side cooldown/last-alert state.
    """
    _SESSION_STATE.pop(session_id, None)
    return {"ok": True, "session_id": session_id}

@app.post("/api/monitor/predict_window")
def predict_window(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Score one window from the live monitor UI.

    This FIXED implementation:
      - Runs *real* checkpoint inference (best.pt) for the selected dataset + arch.
      - Uses per-model operating points (tau_low/tau_high) from outputs/reports.
      - Supports MC Dropout (mu/sigma).
      - Hybrid fusion matches requested policy:
          - NOT_FALL only if both NOT_FALL
          - FALL if (FALL + FALL) or (FALL + UNCERTAIN)
          - otherwise UNCERTAIN
    """
    global LAST_PRED_LATENCY_MS
    t0 = time.time()

    # -------------
    # Inputs
    # -------------
    session_id = str(payload.get("session_id") or "default")

    mode = str(payload.get("mode") or "hybrid").lower().strip()
    if mode in {"hyb", "hybrid", "dual"}:
        mode = "dual"
    elif mode not in {"tcn", "gcn"}:
        # default
        mode = "dual"

    # Dataset selection (payload overrides DB)
    dataset_code = str(payload.get("dataset_code") or payload.get("dataset") or "").lower().strip()

    # Operating point selection (payload overrides DB)
    op_code = str(payload.get("op_code") or payload.get("op") or "").upper().strip()

    # MC dropout selection (payload overrides DB)
    use_mc = payload.get("use_mc")
    mc_M = payload.get("mc_M")

    # Persist events (optional)
    persist = bool(payload.get("persist", False))

    # Best-practice: accept variable-FPS inputs (raw_* + timestamps) and resample here.
    target_T = int(payload.get("target_T") or 48)
    raw_xy = payload.get("raw_xy")
    raw_conf = payload.get("raw_conf")
    raw_t_ms = payload.get("raw_t_ms")
    window_end_t_ms = payload.get("window_end_t_ms", None)

    xy: List[Any] = []
    conf: List[Any] = []
    cap_fps_est: Optional[float] = None

    # -------------
    # Defaults from DB (if available)
    # -------------
    resident_id = int(payload.get("resident_id") or 1)
    cooldown_sec = 3
    active_model_code = "HYBRID" if mode == "dual" else mode.upper()

    try:
        with get_conn() as conn:
            _ensure_system_settings_schema(conn)
            variants = _detect_variants(conn)

            # Read system settings row if available
            sys_row = None
            with conn.cursor() as cur:
                if variants.get("settings") == "v2" and _table_exists(conn, "system_settings"):
                    cur.execute("SELECT * FROM system_settings WHERE resident_id=%s LIMIT 1", (resident_id,))
                    sys_row = cur.fetchone()
                elif _table_exists(conn, "settings"):
                    cur.execute("SELECT * FROM settings WHERE resident_id=%s LIMIT 1", (resident_id,))
                    sys_row = cur.fetchone()

            if isinstance(sys_row, dict):
                if not dataset_code and sys_row.get("active_dataset_code"):
                    dataset_code = str(sys_row.get("active_dataset_code") or "").lower()
                if use_mc is None and sys_row.get("mc_enabled") is not None:
                    use_mc = bool(int(sys_row.get("mc_enabled"))) if str(sys_row.get("mc_enabled")).isdigit() else bool(sys_row.get("mc_enabled"))
                if mc_M is None and sys_row.get("mc_M") is not None:
                    mc_M = int(sys_row.get("mc_M"))
                if sys_row.get("alert_cooldown_sec") is not None:
                    cooldown_sec = int(sys_row.get("alert_cooldown_sec"))
                if sys_row.get("active_model_code"):
                    active_model_code = str(sys_row.get("active_model_code") or active_model_code)

                # derive op_code from active operating point id if present
                op_id = None
                for k in ("active_operating_point", "active_operating_point_id"):
                    if sys_row.get(k) is not None:
                        op_id = sys_row.get(k)
                        break
                if (not op_code) and op_id and _table_exists(conn, "operating_points"):
                    with conn.cursor() as cur:
                        cur.execute("SELECT code FROM operating_points WHERE id=%s LIMIT 1", (int(op_id),))
                        r = cur.fetchone() or {}
                        if isinstance(r, dict) and r.get("code"):
                            op_code = str(r.get("code") or "").upper()
    except Exception:
        # DB is optional for this endpoint
        pass

    if not dataset_code:
        dataset_code = "muvim"
    if not op_code:
        op_code = "OP-2"
    if use_mc is None:
        use_mc = True
    if mc_M is None:
        mc_M = 10

    # Expected FPS per dataset (fallback) — real checkpoints may override internally.
    expected_fps = {
        "le2i": 25,
        "urfd": 30,
        "caucafall": 23,
        "muvim": 30,
    }.get(dataset_code, int(payload.get("target_fps") or payload.get("fps") or 30))

    # Resample if raw inputs exist
    if raw_xy is not None and raw_t_ms is not None:
        xy, conf, _, _, cap_fps_est = _resample_pose_window(
            raw_t_ms=raw_t_ms,
            raw_xy=raw_xy,
            raw_conf=raw_conf,
            target_fps=float(expected_fps),
            target_T=target_T,
            window_end_t_ms=float(window_end_t_ms) if window_end_t_ms is not None else None,
        )

    # Backward-compatible fallback: accept already-resampled windows
    if not xy:
        xy = payload.get("xy") or []
        conf = payload.get("conf") or []

    if not xy:
        raise HTTPException(status_code=400, detail="payload must include raw_* (preferred) or xy")

    # -------------
    # Inference
    # -------------
    specs = _get_deploy_specs()

    def spec_key_for(arch: str) -> str:
        return f"{dataset_code}_{arch}".lower()

    tcn_key = spec_key_for("tcn")
    gcn_key = spec_key_for("gcn")
    if tcn_key not in specs or gcn_key not in specs:
        raise HTTPException(status_code=404, detail=f"No deploy specs found for dataset '{dataset_code}'.")

    models_out: Dict[str, Any] = {}
    tri_tcn = None
    tri_gcn = None

    if mode in {"tcn", "dual"}:
        out_tcn = _predict_spec(
            spec_key=tcn_key,
            xy=xy,
            conf=conf,
            fps=float(expected_fps),
            target_T=target_T,
            op_code=op_code,
            use_mc=bool(use_mc),
            mc_M=int(mc_M),
        )
        models_out["tcn"] = out_tcn
        tri_tcn = out_tcn.get("triage", {}).get("state")

    if mode in {"gcn", "dual"}:
        out_gcn = _predict_spec(
            spec_key=gcn_key,
            xy=xy,
            conf=conf,
            fps=float(expected_fps),
            target_T=target_T,
            op_code=op_code,
            use_mc=bool(use_mc),
            mc_M=int(mc_M),
        )
        models_out["gcn"] = out_gcn
        tri_gcn = out_gcn.get("triage", {}).get("state")

    # Top-level triage decision
    if mode == "tcn":
        triage_state = tri_tcn or "not_fall"
        p_display = float(models_out.get("tcn", {}).get("mu", 0.0))
    elif mode == "gcn":
        triage_state = tri_gcn or "not_fall"
        p_display = float(models_out.get("gcn", {}).get("mu", 0.0))
    else:
        triage_state = _fuse_hybrid(str(tri_tcn), str(tri_gcn))
        mu_t = float(models_out.get("tcn", {}).get("mu", 0.0))
        mu_g = float(models_out.get("gcn", {}).get("mu", 0.0))
        sig_t = float(models_out.get("tcn", {}).get("sigma", 0.0))
        sig_g = float(models_out.get("gcn", {}).get("sigma", 0.0))
        models_out["hybrid"] = {
            "mu": float(min(mu_t, mu_g)),
            "sigma": float(max(sig_t, sig_g)),
            "triage": {"state": triage_state},
            "components": {"tcn": tri_tcn, "gcn": tri_gcn},
        }
        p_display = float(min(mu_t, mu_g))

    # -------------
    # Optional event persistence w/ cooldown
    # -------------
    saved_event_id = None
    if persist and triage_state in {"fall", "uncertain"}:
        now = time.time()
        st = _SESSION_STATE.setdefault(session_id, {})
        last_ts = float(st.get("last_event_ts") or 0.0)
        if now - last_ts >= float(cooldown_sec):
            st["last_event_ts"] = now
            meta = {
                "dataset": dataset_code,
                "mode": mode,
                "op_code": op_code,
                "use_mc": bool(use_mc),
                "mc_M": int(mc_M),
                "expected_fps": expected_fps,
                "capture_fps_est": cap_fps_est,
                "models": models_out,
            }
            try:
                with get_conn() as conn:
                    if _table_exists(conn, "events"):
                        with conn.cursor() as cur:
                            cur.execute(
                                "INSERT INTO events (resident_id, type, severity, model_code, operating_point_id, score, meta) "
                                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                                (
                                    resident_id,
                                    triage_state,
                                    "high" if triage_state == "fall" else "medium",
                                    active_model_code,
                                    None,
                                    float(p_display),
                                    json.dumps(meta),
                                ),
                            )
                            saved_event_id = cur.lastrowid
                        conn.commit()
            except Exception:
                pass

    LAST_PRED_LATENCY_MS = int((time.time() - t0) * 1000)

    return {
        "triage_state": triage_state,
        "models": models_out,
        "latency_ms": LAST_PRED_LATENCY_MS,
        "capture_fps_est": cap_fps_est,
        "target_fps": expected_fps,
        "target_T": target_T,
        "dataset_code": dataset_code,
        "op_code": op_code,
        "use_mc": bool(use_mc),
        "event_id": saved_event_id,
    }

