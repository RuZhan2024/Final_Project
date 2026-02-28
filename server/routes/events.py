from __future__ import annotations

import json
import logging
import uuid

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Body, HTTPException

from ..core import (
    SkeletonClipPayload,
    _anonymize_xy_inplace,
    _cols,
    _detect_variants,
    _event_clips_dir,
    _event_prob_col,
    _event_time_col,
    _has_col,
    _jsonable,
    _one_resident_id,
    _read_clip_privacy_flags,
    _resident_exists,
)
from ..db import get_conn_optional


router = APIRouter()
logger = logging.getLogger(__name__)

_ALLOWED_EVENT_MODELS = {"TCN", "GCN", "HYBRID"}
_ALLOWED_EVENT_STATUSES = {
    "pending_review",
    "unreviewed",
    "confirmed_fall",
    "false_alarm",
    "false_positive",
    "dismissed",
}
_MAX_CLIP_FRAMES = 1200
_MAX_CLIP_JOINTS = 64


@router.get("/api/events")
def list_events(
    resident_id: Optional[int] = None,
    page: int = 1,
    page_size: int = 50,
    start_date: Optional[str] = None,  # YYYY-MM-DD (local UI date)
    end_date: Optional[str] = None,    # YYYY-MM-DD (local UI date, inclusive)
    event_type: Optional[str] = None,  # exact type (e.g., "fall", "uncertain"), or None/"All"
    status: Optional[str] = None,      # pending_review/confirmed_fall/false_alarm/dismissed, or None/"All"
    model: Optional[str] = None,       # GCN/TCN/HYBRID, or None/"All"
    limit: Optional[int] = None,       # legacy: /api/events?limit=500
) -> Dict[str, Any]:
    """List events with server-side pagination (+ optional filters).

    Backward compatible:
      - If the caller uses `limit` (old client), we treat it as `page_size` for page 1.
    """
    # Basic guardrails
    try:
        page = int(page)
    except Exception:
        page = 1
    page = max(1, page)

    # page_size: prefer explicit page_size; but accept legacy `limit` for page 1
    try:
        page_size = int(page_size)
    except Exception:
        page_size = 50

    if limit is not None and page == 1 and page_size == 50:
        # Old clients pass limit=500; keep behaviour similar but cap at 200.
        try:
            page_size = int(limit)
        except Exception:
            pass

    page_size = min(200, max(1, page_size))
    offset = (page - 1) * page_size

    # Normalize filter params ("All" from UI => None)
    def _norm(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        v = str(s).strip()
        if not v or v.lower() == "all":
            return None
        return v

    start_date = _norm(start_date)
    end_date = _norm(end_date)
    event_type = _norm(event_type)
    status = _norm(status)
    model = _norm(model)

    if model is not None:
        model_u = str(model).upper()
        if model_u not in _ALLOWED_EVENT_MODELS:
            raise HTTPException(status_code=400, detail="model must be one of: TCN, GCN, HYBRID")
        model = model_u

    if status is not None:
        status_l = str(status).lower()
        if status_l not in _ALLOWED_EVENT_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "status must be one of: "
                    "pending_review, unreviewed, confirmed_fall, false_alarm, false_positive, dismissed"
                ),
            )
        status = status_l

    # Convert date-only strings to datetime bounds (inclusive end_date by using < next_day)
    start_dt = None
    end_excl_dt = None
    try:
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
    except Exception:
        start_dt = None
    try:
        if end_date:
            end_excl_dt = datetime.fromisoformat(end_date) + timedelta(days=1)
    except Exception:
        end_excl_dt = None

    with get_conn_optional() as conn:
        if conn is None:
            rid = int(resident_id or 1)
            return {
                "resident_id": rid,
                "events": [],
                "total": 0,
                "page": page,
                "page_size": page_size,
                "db_available": False,
            }

        rid = resident_id if resident_id and _resident_exists(conn, resident_id) else _one_resident_id(conn)
        variants = _detect_variants(conn)
        time_col = _event_time_col(conn)
        prob_col = _event_prob_col(conn)

        with conn.cursor() as cur:
            # Build WHERE clause (shared)
            where = ["e.resident_id=%s"]
            params: List[Any] = [rid]

            if start_dt is not None:
                where.append(f"e.`{time_col}` >= %s")
                params.append(start_dt)
            if end_excl_dt is not None:
                where.append(f"e.`{time_col}` < %s")
                params.append(end_excl_dt)
            if event_type is not None:
                where.append("LOWER(e.`type`)=%s")
                params.append(event_type.lower())

            if variants["events"] == "v2":
                # v2 has explicit status field
                if status is not None:
                    where.append("LOWER(e.`status`)=%s")
                    params.append(status.lower())

                # v2 model filter uses models table
                join_models = "LEFT JOIN models m ON m.id = e.model_id"
                if model is not None:
                    where.append("(UPPER(m.code)=%s OR UPPER(m.family)=%s)")
                    u = model.upper()
                    params.extend([u, u])

                where_sql = " AND ".join(where)

                # Total count (for pagination UI)
                cur.execute(
                    f"""SELECT COUNT(*) AS n
                          FROM events e
                          {join_models}
                          WHERE {where_sql}""",
                    tuple(params),
                )
                total = int((cur.fetchone() or {}).get("n") or 0)

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
                          {join_models}
                          WHERE {where_sql}
                          ORDER BY e.`{time_col}` DESC
                          LIMIT %s OFFSET %s""",
                    tuple(params + [page_size, offset]),
                )
                rows = cur.fetchall() or []

                out: List[Dict[str, Any]] = []
                for r in rows:
                    meta: Dict[str, Any] = {}
                    if r.get("notes") is not None:
                        meta["notes"] = r.get("notes")
                    if r.get("fa24h_snapshot") is not None:
                        meta["fa24h_snapshot"] = r.get("fa24h_snapshot")
                    if r.get("payload_json") is not None:
                        meta["payload_json"] = r.get("payload_json")

                    ts = r.get("ts")
                    out.append(
                        {
                            "id": r.get("id"),
                            # Keep both keys for front-end compatibility
                            "event_time": ts,
                            "ts": ts,
                            "type": r.get("type"),
                            "status": r.get("status"),
                            "score": r.get("score"),
                            "p_fall": r.get("score"),
                            "model_code": r.get("model_code") or r.get("model_family"),
                            "operating_point_id": r.get("operating_point_id"),
                            "meta": meta,
                        }
                    )

                return _jsonable(
                    {
                        "resident_id": rid,
                        "events": out,
                        "total": total,
                        "page": page,
                        "page_size": page_size,
                    }
                )

            # ---- v1 events schema fallback ----
            # v1 may not support status; we only filter by model_code if available.
            if model is not None:
                where.append("UPPER(e.model_code)=%s")
                params.append(model.upper())

            where_sql = " AND ".join(where)

            cur.execute(
                f"""SELECT COUNT(*) AS n
                      FROM events e
                      WHERE {where_sql}""",
                tuple(params),
            )
            total = int((cur.fetchone() or {}).get("n") or 0)

            prob_select = f"e.`{prob_col}` AS score," if prob_col else "NULL AS score,"

            cur.execute(
                f"""SELECT e.id,
                             e.`{time_col}` AS ts,
                             e.`type` AS type,
                             e.severity,
                             e.model_code,
                             {prob_select}
                             e.meta
                      FROM events e
                      WHERE {where_sql}
                      ORDER BY e.`{time_col}` DESC
                      LIMIT %s OFFSET %s""",
                tuple(params + [page_size, offset]),
            )

            rows = cur.fetchall() or []

            out: List[Dict[str, Any]] = []
            for r in rows:
                meta = r.get("meta")
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {"raw": meta}

                ts = r.get("ts")
                out.append(
                    {
                        "id": r.get("id"),
                        "event_time": ts,
                        "ts": ts,
                        "type": r.get("type"),
                        "severity": r.get("severity"),
                        "model_code": r.get("model_code"),
                        "score": r.get("score"),
                            "p_fall": r.get("score"),
                        "meta": meta,
                    }
                )

            return _jsonable(
                {
                    "resident_id": rid,
                    "events": out,
                    "total": total,
                    "page": page,
                    "page_size": page_size,
                }
            )


@router.get("/api/events/summary")
def events_summary(resident_id: Optional[int] = None) -> Dict[str, Any]:
    with get_conn_optional() as conn:
        if conn is None:
            rid = int(resident_id or 1)
            return {
                "resident_id": rid,
                "total_events": 0,
                "total_falls": 0,
                "events_last_24h": 0,
                "falls_last_24h": 0,
                "latest_event": None,
                "today": {"falls": 0, "pending": 0, "false_alarms": 0},
                "db_available": False,
            }
        rid = resident_id if resident_id and _resident_exists(conn, resident_id) else _one_resident_id(conn)
        time_col = _event_time_col(conn)
        event_cols = _cols(conn, "events")

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

                pending_24h = 0
                false_alarms_24h = 0
                if "status" in event_cols:
                    cur.execute(
                        f"""
                        SELECT COUNT(*) AS n
                        FROM events
                        WHERE resident_id=%s
                          AND `{time_col}` >= %s
                          AND LOWER(`status`) IN ('pending_review', 'unreviewed')
                        """,
                        (rid, since),
                    )
                    pending_24h = int((cur.fetchone() or {}).get("n") or 0)

                    cur.execute(
                        f"""
                        SELECT COUNT(*) AS n
                        FROM events
                        WHERE resident_id=%s
                          AND `{time_col}` >= %s
                          AND LOWER(`status`) IN ('false_alarm', 'false_positive')
                        """,
                        (rid, since),
                    )
                    false_alarms_24h = int((cur.fetchone() or {}).get("n") or 0)
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
            "today": {
                "falls": falls_24h,
                "pending": pending_24h,
                "false_alarms": false_alarms_24h,
            },
            "db_available": True,
        }
    )


@router.post("/api/events/test_fall")
def test_fall() -> Dict[str, Any]:
    """Insert a synthetic 'fall' event for UI testing."""
    with get_conn_optional() as conn:
        if conn is None:
            return {"ok": False, "reason": "db_unavailable"}
        rid = _one_resident_id(conn)
        variants = _detect_variants(conn)
        now = datetime.utcnow()

        with conn.cursor() as cur:
            if variants["events"] == "v2":
                model_id = None
                op_id = None
                if _has_col(conn, "system_settings", "active_model_id"):
                    cur.execute(
                        "SELECT active_model_id, active_operating_point_id FROM system_settings WHERE resident_id=%s LIMIT 1",
                        (rid,),
                    )
                    s = cur.fetchone() or {}
                    model_id = s.get("active_model_id")
                    op_id = s.get("active_operating_point_id")

                cols = _cols(conn, "events")
                insert_cols = ["resident_id"]
                insert_vals = ["%s"]
                params: List[Any] = [rid]

                def add(col: str, val: Any) -> None:
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

                sql = (
                    f"INSERT INTO events ({', '.join('`'+c+'`' for c in insert_cols)}) "
                    f"VALUES ({', '.join(insert_vals)})"
                )
                cur.execute(sql, tuple(params))
                new_id = cur.lastrowid
                cur.execute("SELECT * FROM events WHERE id=%s", (new_id,))
                row = cur.fetchone()
                return {"ok": True, "event": _jsonable(row)}

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


@router.post("/api/events/{event_id}/skeleton_clip")
def upload_skeleton_clip(event_id: int, payload: SkeletonClipPayload = Body(...)) -> Dict[str, Any]:
    """Persist a short skeleton-only clip and attach it to an existing event."""
    rid = int(payload.resident_id or 1)

    with get_conn_optional() as conn:
        if conn is None:
            raise HTTPException(status_code=503, detail="DB not available")
        store_event_clips, anonymize = _read_clip_privacy_flags(conn, rid)
        if not store_event_clips:
            return {"ok": True, "skipped": True, "reason": "store_event_clips_disabled"}

        event_cols = _cols(conn, "events")
        meta_col = "meta" if "meta" in event_cols else ("payload_json" if "payload_json" in event_cols else None)

        with conn.cursor() as cur:
            select_cols = ["id", "resident_id"]
            if meta_col:
                select_cols.append(meta_col)
            cur.execute(
                f"SELECT {', '.join('`'+c+'`' for c in select_cols)} FROM events WHERE id=%s LIMIT 1",
                (int(event_id),),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
        if int(row.get("resident_id") or 0) != rid:
            raise HTTPException(status_code=403, detail="Event does not belong to resident")

        t_ms = np.asarray(payload.t_ms, dtype=np.float32)
        if t_ms.ndim != 1 or t_ms.size < 2:
            raise HTTPException(status_code=400, detail="t_ms must be a 1D array with at least 2 timestamps")
        if t_ms.size > _MAX_CLIP_FRAMES:
            raise HTTPException(status_code=413, detail=f"clip too long: max {_MAX_CLIP_FRAMES} frames")
        if not np.all(np.isfinite(t_ms)):
            raise HTTPException(status_code=400, detail="t_ms contains non-finite values")
        if payload.xy is not None:
            xy = np.asarray(payload.xy, dtype=np.float32)
            if xy.ndim != 3 or xy.shape[-1] != 2:
                raise HTTPException(status_code=400, detail="xy must be shaped [T,J,2]")
        elif payload.xy_flat is not None:
            xy_flat = np.asarray(payload.xy_flat, dtype=np.float32).reshape(-1)
            n_joints = int(payload.raw_joints) if payload.raw_joints is not None else 33
            if n_joints < 1 or n_joints > _MAX_CLIP_JOINTS:
                raise HTTPException(status_code=400, detail=f"raw_joints must be in [1, {_MAX_CLIP_JOINTS}]")
            t_len = int(t_ms.shape[0])
            expected = int(t_len * n_joints * 2)
            if xy_flat.size != expected:
                raise HTTPException(status_code=400, detail="xy_flat size mismatch for [T,J,2]")
            xy = xy_flat.reshape(t_len, n_joints, 2)
        else:
            raise HTTPException(status_code=400, detail="provide either xy or xy_flat")
        T = int(xy.shape[0])
        if T < 2:
            raise HTTPException(status_code=400, detail="clip must contain at least 2 frames")
        if T > _MAX_CLIP_FRAMES:
            raise HTTPException(status_code=413, detail=f"clip too long: max {_MAX_CLIP_FRAMES} frames")
        if int(xy.shape[1]) < 1 or int(xy.shape[1]) > _MAX_CLIP_JOINTS:
            raise HTTPException(status_code=400, detail=f"joint dimension must be in [1, {_MAX_CLIP_JOINTS}]")
        if t_ms.shape[0] != T:
            raise HTTPException(status_code=400, detail="t_ms and xy time dimensions must match")
        if not bool(np.isfinite(xy).all()):
            raise HTTPException(status_code=400, detail="xy contains non-finite values")
        if t_ms.size > 1 and not bool(np.all(t_ms[1:] >= t_ms[:-1])):
            raise HTTPException(status_code=400, detail="t_ms must be monotonically non-decreasing")

        if payload.conf is not None:
            conf = np.asarray(payload.conf, dtype=np.float32)
            if conf.ndim != 2:
                raise HTTPException(status_code=400, detail="conf must be shaped [T,J]")
            if conf.shape[0] != T:
                raise HTTPException(status_code=400, detail="conf and xy time dimensions must match")
        elif payload.conf_flat is not None:
            conf_flat = np.asarray(payload.conf_flat, dtype=np.float32).reshape(-1)
            expected = int(T * xy.shape[1])
            if conf_flat.size != expected:
                raise HTTPException(status_code=400, detail="conf_flat size mismatch for [T,J]")
            conf = conf_flat.reshape(T, xy.shape[1])
        else:
            conf = np.ones((T, xy.shape[1]), dtype=np.float32)
        if conf.shape[1] != xy.shape[1]:
            raise HTTPException(status_code=400, detail="conf and xy joint dimensions must match")
        if not bool(np.isfinite(conf).all()):
            raise HTTPException(status_code=400, detail="conf contains non-finite values")
        np.clip(xy, 0.0, 1.0, out=xy)
        np.clip(conf, 0.0, 1.0, out=conf)

        if anonymize:
            xy = _anonymize_xy_inplace(xy)
            np.around(xy, 4, out=xy)
            np.around(conf, 4, out=conf)

        clips_dir = _event_clips_dir()
        fname = f"event_{int(event_id)}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.npz"
        fpath = clips_dir / fname
        try:
            np.savez_compressed(fpath, t_ms=t_ms, xy=xy, conf=conf)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save clip: {e}")

        clip_rel = f"server/event_clips/{fname}"

        meta = row.get(meta_col) if meta_col else {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        meta["skeleton_clip"] = {
            "path": clip_rel,
            "n_frames": int(T),
            "n_joints": int(xy.shape[1]),
            "pre_s": float(payload.pre_s) if payload.pre_s is not None else None,
            "post_s": float(payload.post_s) if payload.post_s is not None else None,
            "dataset_code": (payload.dataset_code or None),
            "mode": (payload.mode or None),
            "op_code": (payload.op_code or None),
            "use_mc": bool(payload.use_mc) if payload.use_mc is not None else None,
            "mc_M": int(payload.mc_M) if payload.mc_M is not None else None,
            "mc_sigma_tol": float(payload.mc_sigma_tol) if payload.mc_sigma_tol is not None else None,
            "mc_se_tol": float(payload.mc_se_tol) if payload.mc_se_tol is not None else None,
            "anonymized": bool(anonymize),
            "saved_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        }

        if meta_col:
            try:
                with conn.cursor() as cur:
                    try:
                        meta_json = json.dumps(meta, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
                    except Exception:
                        meta_json = json.dumps(meta, separators=(",", ":"), ensure_ascii=False)
                    cur.execute(
                        f"UPDATE events SET `{meta_col}`=%s WHERE id=%s",
                        (meta_json, int(event_id)),
                    )
                conn.commit()
            except Exception:
                logger.warning("Failed to update event %s with skeleton_clip for event_id=%s", meta_col, event_id, exc_info=True)
        else:
            logger.warning("events table has no `meta`/`payload_json`; skeleton clip metadata not persisted for event_id=%s", event_id)

        return {
            "ok": True,
            "event_id": int(event_id),
            "clip_path": clip_rel,
            "n_frames": int(T),
            "anonymized": bool(anonymize),
        }
