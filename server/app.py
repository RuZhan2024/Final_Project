# server/app.py
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Support running as package (uvicorn server.app:app) or from cwd (uvicorn app:app)
try:
    from .db import get_conn
except Exception:  # pragma: no cover
    from db import get_conn


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
    # v1-style fields (some may not exist in DB; we store what we can)
    monitoring_enabled: Optional[bool] = None
    api_online: Optional[bool] = None
    alert_cooldown_sec: Optional[int] = None
    notify_on_every_fall: Optional[bool] = None

    # model selection
    active_model_code: Optional[str] = Field(default=None, description="TCN | GCN | HYBRID (or any code present in DB)")
    active_operating_point: Optional[int] = Field(default=None, description="operating_points.id")

    # v2-style fields
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
def get_settings(resident_id: Optional[int] = None) -> Dict[str, Any]:
    with get_conn() as conn:
        rid = resident_id if resident_id and _resident_exists(conn, resident_id) else _one_resident_id(conn)
        variants = _detect_variants(conn)

        with conn.cursor() as cur:
            if variants["settings"] == "v2":
                cur.execute("SELECT * FROM system_settings WHERE resident_id=%s LIMIT 1", (rid,))
                srow = cur.fetchone() or {}
                active_model_id = srow.get("active_model_id")
                active_op_id = srow.get("active_operating_point_id")

                out = {
                    "resident_id": rid,
                    "risk_profile": srow.get("risk_profile"),
                    "notify_email": srow.get("notify_email"),
                    "notify_sms": bool(srow.get("notify_sms")) if srow.get("notify_sms") is not None else None,
                    "active_model_id": active_model_id,
                    "active_model_code": _resolve_model_code(conn, active_model_id),
                    "active_operating_point_id": active_op_id,
                    "active_operating_point": active_op_id,
                }
                return _jsonable(out)

            # v1
            cur.execute("SELECT * FROM system_settings LIMIT 1")
            srow = cur.fetchone() or {}
            return _jsonable(srow)


@app.put("/api/settings")
def update_settings(payload: SettingsUpdatePayload) -> Dict[str, Any]:
    with get_conn() as conn:
        variants = _detect_variants(conn)

        if variants["settings"] == "v2":
            rid = _one_resident_id(conn)
            active_model_id = None
            if payload.active_model_code:
                active_model_id = _resolve_model_id(conn, payload.active_model_code)

            active_op_id = payload.active_operating_point
            if active_op_id is not None:
                active_op_id = _resolve_op_id(conn, active_model_id, active_op_id)

            # Build UPDATE dynamically based on available columns
            updates: List[str] = []
            params: List[Any] = []

            cols = _cols(conn, "system_settings")
            if "active_model_id" in cols and active_model_id is not None:
                updates.append("active_model_id=%s")
                params.append(active_model_id)
            if "active_operating_point_id" in cols and active_op_id is not None:
                updates.append("active_operating_point_id=%s")
                params.append(active_op_id)
            if payload.risk_profile is not None and "risk_profile" in cols:
                updates.append("risk_profile=%s")
                params.append(payload.risk_profile)
            if payload.notify_email is not None and "notify_email" in cols:
                updates.append("notify_email=%s")
                params.append(payload.notify_email)
            if payload.notify_sms is not None and "notify_sms" in cols:
                updates.append("notify_sms=%s")
                params.append(int(payload.notify_sms))

            if not updates:
                # Nothing to update, just return current settings
                return get_settings(rid)

            params.append(rid)
            sql = "UPDATE system_settings SET " + ", ".join(updates) + " WHERE resident_id=%s"
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))

            return get_settings(rid)

        # v1
        # Expect a single row system_settings. Update only columns that exist.
        cols = _cols(conn, "system_settings")
        updates: List[str] = []
        params: List[Any] = []

        def _set(col: str, val: Any):
            if col in cols and val is not None:
                updates.append(f"{col}=%s")
                params.append(val)

        _set("monitoring_enabled", payload.monitoring_enabled)
        _set("api_online", payload.api_online)
        _set("active_model_code", payload.active_model_code)
        _set("active_operating_point", payload.active_operating_point)
        _set("alert_cooldown_sec", payload.alert_cooldown_sec)
        _set("notify_on_every_fall", payload.notify_on_every_fall)

        if updates:
            with conn.cursor() as cur:
                cur.execute("UPDATE system_settings SET " + ", ".join(updates) + " WHERE id=1", tuple(params))

        return get_settings(None)


# -----------------------------
# Events
# -----------------------------
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
def dashboard_summary() -> Dict[str, Any]:
    with get_conn() as conn:
        rid = _one_resident_id(conn)
        # Use events_summary + settings + last 20 events
        evsum = events_summary(rid)
        settings = get_settings(rid)
        recent = list_events(rid, limit=20)

    return {
        "resident_id": rid,
        "settings": settings,
        "events_summary": evsum,
        "recent_events": recent.get("events", []),
    }


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
    """Predict on one window sent from /monitor-demo.

    The frontend can send different payload shapes depending on version.
    This handler is intentionally tolerant: it will read any JSON, and return a
    consistent response schema for the UI.

    Replace the heuristic part with real TCN/GCN/HYBRID inference when ready.
    """
    model_code = str(payload.get("model_code") or payload.get("model") or "HYBRID").upper()
    resident_id = int(payload.get("resident_id") or 1)

    window = payload.get("window") or payload.get("data") or payload

    # Simple heuristic probability (keeps demo functional end-to-end)
    hint = str(window.get("hint") or window.get("label") or "").lower() if isinstance(window, dict) else ""
    p = 0.10
    if "fall" in hint:
        p = 0.92
    elif "uncertain" in hint:
        p = 0.55

    # Pull thresholds from currently active operating point (if available)
    thr_low, thr_high = 0.65, 0.90
    try:
        with get_conn() as conn:
            variants = _detect_variants(conn)
            with conn.cursor() as cur:
                if variants.get("settings") == "v2":
                    cur.execute("SELECT active_model_id, active_operating_point FROM system_settings WHERE resident_id=%s", (resident_id,))
                    s = cur.fetchone()
                    op_id = s.get("active_operating_point") if s else None
                    if op_id:
                        if variants.get("ops") == "v2":
                            cur.execute("SELECT thr_low_conf, thr_high_conf FROM operating_points WHERE id=%s", (op_id,))
                            r = cur.fetchone() or {}
                            if r.get("thr_low_conf") is not None:
                                thr_low = float(r["thr_low_conf"])
                            if r.get("thr_high_conf") is not None:
                                thr_high = float(r["thr_high_conf"])
                        else:
                            cur.execute("SELECT threshold_low, threshold_high FROM operating_points WHERE id=%s", (op_id,))
                            r = cur.fetchone() or {}
                            if r.get("threshold_low") is not None:
                                thr_low = float(r["threshold_low"])
                            if r.get("threshold_high") is not None:
                                thr_high = float(r["threshold_high"])
    except Exception:
        pass

    # 3-way decision
    if p >= thr_high:
        decision = "fall"
    elif p >= thr_low:
        decision = "uncertain"
    else:
        decision = "not_fall"

    return {
        "model_code": model_code,
        "resident_id": resident_id,
        "p_fall": float(p),
        "decision": decision,
        "threshold_low": float(thr_low),
        "threshold_high": float(thr_high),
    }
