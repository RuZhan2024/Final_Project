#!/usr/bin/env python3
"""
FastAPI backend for Elder Fall Monitor demo.

Key endpoints used by the React app:
  - GET  /api/settings
  - PUT  /api/settings
  - GET  /api/operating_points?model_code=GCN
  - POST /api/events/test_fall
  - GET  /api/dashboard/summary
  - GET  /api/events
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .db import get_conn

app = FastAPI(title="Elder Fall Monitor API", version="0.2.0")

# CORS for local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Pydantic payloads
# -----------------------------
class SettingsUpdatePayload(BaseModel):
    monitoring_enabled: Optional[bool] = None
    api_online: Optional[bool] = None
    active_model_code: Optional[str] = Field(default=None, description="TCN | GCN | HYBRID")
    active_operating_point: Optional[int] = Field(default=None, description="operating_points.id")
    alert_cooldown_sec: Optional[int] = None
    notify_on_every_fall: Optional[bool] = None


# -----------------------------
# Helpers
# -----------------------------
def _one_resident_id(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM residents ORDER BY id ASC LIMIT 1")
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=500, detail="No residents in DB. Run create_db_fixed.sql seed inserts.")
        return int(row["id"])


def _ensure_system_settings_row(conn, resident_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM system_settings WHERE resident_id=%s LIMIT 1", (resident_id,))
        row = cur.fetchone()
        if row:
            return
        cur.execute(
            """
            INSERT INTO system_settings
              (resident_id, monitoring_enabled, api_online, last_latency_ms, active_model_code,
               active_operating_point, alert_cooldown_sec, notify_on_every_fall)
            VALUES
              (%s, 1, 1, NULL, 'GCN', NULL, 3, 1)
            """,
            (resident_id,),
        )


def _get_models_map(conn) -> Dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, code FROM models")
        rows = cur.fetchall() or []
    return {str(r["code"]).upper(): int(r["id"]) for r in rows}


def _bootstrap_ops_if_empty(conn, model_id: int) -> None:
    """Create 3 default operating points if none exist for model_id."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM operating_points WHERE model_id=%s", (model_id,))
        n = int((cur.fetchone() or {}).get("n") or 0)
        if n > 0:
            return

        # Default (demo) thresholds
        rows = [
            ("OP-1", "High Sensitivity", 0.35, 0.20, 0.60),
            ("OP-2", "Balanced",         0.50, 0.30, 0.70),
            ("OP-3", "Low Sensitivity",  0.70, 0.50, 0.85),
        ]
        cur.executemany(
            """
            INSERT INTO operating_points
              (model_id, code, name, thr_detect, thr_low_conf, thr_high_conf, est_fa24h, est_recall)
            VALUES
              (%s, %s, %s, %s, %s, %s, NULL, NULL)
            """,
            [(model_id, c, n, td, tl, th) for (c, n, td, tl, th) in rows],
        )


# -----------------------------
# API
# -----------------------------
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ts": datetime.utcnow().isoformat()}


@app.get("/api/settings")
def get_settings() -> Dict[str, Any]:
    with get_conn() as conn:
        resident_id = _one_resident_id(conn)
        _ensure_system_settings_row(conn, resident_id)

        with conn.cursor() as cur:
            cur.execute("SELECT * FROM system_settings WHERE resident_id=%s LIMIT 1", (resident_id,))
            srow = cur.fetchone() or {}

            cur.execute("SELECT * FROM caregivers WHERE resident_id=%s ORDER BY id ASC LIMIT 1", (resident_id,))
            crow = cur.fetchone() or {}

        sys = {
            "monitoring_enabled": bool(srow.get("monitoring_enabled")),
            "api_online": bool(srow.get("api_online")),
            "last_latency_ms": srow.get("last_latency_ms"),
            "active_model_code": srow.get("active_model_code") or "GCN",
            "active_operating_point": srow.get("active_operating_point"),
            "alert_cooldown_sec": srow.get("alert_cooldown_sec") or 3,
            "notify_on_every_fall": bool(srow.get("notify_on_every_fall")),
        }

        caregiver = {
            "name": crow.get("name") or "",
            "email": crow.get("email") or "",
            "phone": crow.get("phone") or "",
        }

        return {"sys": sys, "caregiver": caregiver}


@app.put("/api/settings")
def update_settings(payload: SettingsUpdatePayload) -> Dict[str, Any]:
    with get_conn() as conn:
        resident_id = _one_resident_id(conn)
        _ensure_system_settings_row(conn, resident_id)

        fields: List[str] = []
        vals: List[Any] = []

        if payload.monitoring_enabled is not None:
            fields.append("monitoring_enabled=%s")
            vals.append(1 if payload.monitoring_enabled else 0)

        if payload.api_online is not None:
            fields.append("api_online=%s")
            vals.append(1 if payload.api_online else 0)

        if payload.active_model_code is not None:
            fields.append("active_model_code=%s")
            vals.append(payload.active_model_code.upper())

        if payload.active_operating_point is not None:
            fields.append("active_operating_point=%s")
            vals.append(int(payload.active_operating_point))

        if payload.alert_cooldown_sec is not None:
            fields.append("alert_cooldown_sec=%s")
            vals.append(int(payload.alert_cooldown_sec))

        if payload.notify_on_every_fall is not None:
            fields.append("notify_on_every_fall=%s")
            vals.append(1 if payload.notify_on_every_fall else 0)

        if not fields:
            return {"ok": True}

        vals.append(resident_id)
        q = "UPDATE system_settings SET " + ", ".join(fields) + " WHERE resident_id=%s"
        with conn.cursor() as cur:
            cur.execute(q, vals)

        return {"ok": True}


@app.get("/api/operating_points")
def list_operating_points(model_code: str = "GCN") -> Dict[str, Any]:
    model_code = (model_code or "GCN").upper()

    with get_conn() as conn:
        models = _get_models_map(conn)
        if model_code not in models:
            raise HTTPException(status_code=404, detail=f"Unknown model_code={model_code}. Known: {sorted(models)}")

        model_id = models[model_code]
        _bootstrap_ops_if_empty(conn, model_id)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, code AS op_code, name, thr_detect, thr_low_conf, thr_high_conf, est_fa24h, est_recall
                FROM operating_points
                WHERE model_id=%s
                ORDER BY id ASC
                """,
                (model_id,),
            )
            ops = cur.fetchall() or []

    return {"model_code": model_code, "ops": ops}


@app.get("/api/events")
def list_events(limit: int = 100) -> Dict[str, Any]:
    limit = max(1, min(int(limit), 500))
    with get_conn() as conn:
        resident_id = _one_resident_id(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, ts, type, severity, model_code, operating_point_id, score, meta
                FROM events
                WHERE resident_id=%s
                ORDER BY ts DESC
                LIMIT %s
                """,
                (resident_id, limit),
            )
            rows = cur.fetchall() or []

    # Convert datetimes to ISO strings
    for r in rows:
        if isinstance(r.get("ts"), datetime):
            r["ts"] = r["ts"].isoformat()
        if isinstance(r.get("meta"), (bytes, bytearray)):
            try:
                r["meta"] = json.loads(r["meta"].decode("utf-8"))
            except Exception:
                r["meta"] = None
    return {"events": rows}


@app.post("/api/events/test_fall")
def test_fall() -> Dict[str, Any]:
    with get_conn() as conn:
        resident_id = _one_resident_id(conn)
        _ensure_system_settings_row(conn, resident_id)

        # Get current active model/op
        with conn.cursor() as cur:
            cur.execute("SELECT active_model_code, active_operating_point FROM system_settings WHERE resident_id=%s LIMIT 1", (resident_id,))
            srow = cur.fetchone() or {}
        model_code = (srow.get("active_model_code") or "GCN").upper()
        op_id = srow.get("active_operating_point")

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO events (resident_id, ts, type, severity, model_code, operating_point_id, score, meta)
                VALUES (%s, NOW(), 'fall', 'high', %s, %s, %s, %s)
                """,
                (resident_id, model_code, op_id, 0.95, json.dumps({"source": "test_fall"})),
            )
            event_id = cur.lastrowid

    return {"ok": True, "event_id": event_id}


@app.get("/api/dashboard/summary")
def dashboard_summary() -> Dict[str, Any]:
    with get_conn() as conn:
        resident_id = _one_resident_id(conn)
        now = datetime.utcnow()
        since = now - timedelta(days=1)

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM events WHERE resident_id=%s", (resident_id,))
            total = int((cur.fetchone() or {}).get("n") or 0)

            cur.execute("SELECT COUNT(*) AS n FROM events WHERE resident_id=%s AND type='fall'", (resident_id,))
            falls = int((cur.fetchone() or {}).get("n") or 0)

            cur.execute(
                "SELECT COUNT(*) AS n FROM events WHERE resident_id=%s AND ts >= %s",
                (resident_id, since),
            )
            last_24h = int((cur.fetchone() or {}).get("n") or 0)

            cur.execute(
                """
                SELECT id, ts, type, severity, model_code, operating_point_id, score
                FROM events
                WHERE resident_id=%s
                ORDER BY ts DESC
                LIMIT 1
                """,
                (resident_id,),
            )
            latest = cur.fetchone() or None

    if latest and isinstance(latest.get("ts"), datetime):
        latest["ts"] = latest["ts"].isoformat()

    return {
        "total_events": total,
        "total_falls": falls,
        "events_last_24h": last_24h,
        "latest_event": latest,
    }

@app.get("/api/models/summary")
def models_summary():
    # Minimal, frontend-friendly summary
    modes = []
    for code in ["TCN", "GCN", "HYBRID"]:
        modes.append({
            "model_code": code,
            "name": code,
            "description": {
                "TCN": "Temporal CNN (sequence model)",
                "GCN": "Graph Convolutional Network (skeleton graph)",
                "HYBRID": "Ensemble / combined score (TCN + GCN)"
            }.get(code, code),
            "supports_mc_dropout": True,
            "supports_uncertain": True,
        })

    return {"models": modes}