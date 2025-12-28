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
        variants = _detect_variants(conn)

        # Defaults (safe even if DB has minimal schema)
        system: Dict[str, Any] = {
            "monitoring_enabled": True,
            "active_model_code": "HYBRID",
            "active_operating_point": None,
            "camera_source": "webcam",
            "fall_threshold": 0.85,
            "alert_cooldown_sec": 3,
            "store_event_clips": False,
            "anonymize_skeleton_data": True,
            "require_confirmation": False,
            "notify_on_every_fall": True,
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
                "monitoring_enabled": bool(_safe_get(row, "monitoring_enabled", 1)),
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
                    system["monitoring_enabled"] = bool(sys_row.get("monitoring_enabled", 1))

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
                ]:
                    if isinstance(sys_row, dict) and col in sys_row and sys_row.get(col) is not None:
                        try:
                            setter(sys_row[col])
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
    """Score one window from the live monitor UI and return UI-friendly fields."""
    global LAST_PRED_LATENCY_MS
    t0 = time.time()

    xy = payload.get("xy") or []
    mode = str(payload.get("mode") or "hybrid").lower()
    spec_id = payload.get("spec_id", None)

    def motion_prob(xy_arr: Any) -> float:
        try:
            import math
            if not isinstance(xy_arr, list) or len(xy_arr) < 2:
                return 0.05
            total = 0.0
            cnt = 0
            for t in range(1, len(xy_arr)):
                f = xy_arr[t]
                p = xy_arr[t - 1]
                if not isinstance(f, list) or not isinstance(p, list):
                    continue
                J = min(len(f), len(p))
                for j in range(J):
                    a = f[j]
                    b = p[j]
                    if not (isinstance(a, list) and isinstance(b, list) and len(a) >= 2 and len(b) >= 2):
                        continue
                    dx = float(a[0]) - float(b[0])
                    dy = float(a[1]) - float(b[1])
                    total += math.sqrt(dx * dx + dy * dy)
                    cnt += 1
            if cnt == 0:
                return 0.05
            v = total / cnt
            p = 1.0 / (1.0 + math.exp(-(v - 0.03) / 0.01))
            return max(0.0, min(1.0, p))
        except Exception:
            return 0.05

    base_p = motion_prob(xy)
    p_tcn = max(0.0, min(1.0, base_p * 0.92 + 0.03))
    p_gcn = max(0.0, min(1.0, base_p * 1.00 + 0.02))
    p_hyb = max(p_tcn, p_gcn)

    # Defaults if DB doesn't have operating points
    tau_low = 0.60
    tau_high = 0.85

    # Try to load thresholds from operating_points (spec_id preferred)
    try:
        with get_conn() as conn:
            if _table_exists(conn, "operating_points"):
                op_id = spec_id
                if op_id is None and _table_exists(conn, "system_settings") and _col_exists(conn, "system_settings", "active_operating_point_id"):
                    with conn.cursor() as cur:
                        cur.execute("SELECT active_operating_point_id FROM system_settings WHERE id=1 LIMIT 1")
                        r = cur.fetchone() or {}
                        if isinstance(r, dict):
                            op_id = r.get("active_operating_point_id")
                if op_id is not None:
                    with conn.cursor() as cur:
                        cur.execute("SELECT tau_low, tau_high FROM operating_points WHERE id=%s LIMIT 1", (op_id,))
                        op = cur.fetchone() or {}
                    if isinstance(op, dict):
                        if op.get("tau_low") is not None:
                            tau_low = float(op["tau_low"])
                        if op.get("tau_high") is not None:
                            tau_high = float(op["tau_high"])
    except Exception:
        pass

    def triage_for(p: float) -> str:
        if p >= tau_high:
            return "fall"
        if p >= tau_low:
            return "uncertain"
        return "not_fall"

    models_out = {
        "tcn": {"mu": float(p_tcn), "sigma": float(max(0.02, 0.25 * (1.0 - p_tcn))), "triage": {"tau_low": tau_low, "tau_high": tau_high}},
        "gcn": {"mu": float(p_gcn), "sigma": float(max(0.02, 0.25 * (1.0 - p_gcn))), "triage": {"tau_low": tau_low, "tau_high": tau_high}},
        "hybrid": {"mu": float(p_hyb), "sigma": float(max(0.02, 0.25 * (1.0 - p_hyb))), "triage": {"tau_low": tau_low, "tau_high": tau_high}},
    }

    if mode == "tcn":
        triage_state = triage_for(p_tcn)
    elif mode == "gcn":
        triage_state = triage_for(p_gcn)
    else:
        triage_state = triage_for(p_hyb)

    LAST_PRED_LATENCY_MS = int((time.time() - t0) * 1000)

    return {"triage_state": triage_state, "models": models_out, "latency_ms": LAST_PRED_LATENCY_MS}

