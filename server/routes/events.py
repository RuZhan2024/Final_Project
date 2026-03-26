from __future__ import annotations

import json
import uuid

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Body, HTTPException

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

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
    _table_exists,
)
from ..db import get_conn_optional
from ..notifications import get_notification_manager
from ..notifications.models import NotificationPreferences, SafeGuardEvent
from ..notifications_service import dispatch_fall_notifications


router = APIRouter()


def _dispatch_safe_guard_from_event(
    conn: Any,
    *,
    resident_id: int,
    event_id: int,
    model_code: str,
    dataset_code: str = "caucafall",
    op_code: str = "OP-2",
    p_fall: float = 0.99,
    location: str = "events_test_fall",
) -> None:
    notify_on_every_fall = True
    notify_sms = False
    notify_phone = False
    caregiver_name = ""
    caregiver_email = ""
    caregiver_phone = ""

    if _table_exists(conn, "system_settings"):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM system_settings WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
                (int(resident_id),),
            )
            s = cur.fetchone() or {}
        if isinstance(s, dict):
            notify_on_every_fall = bool(s.get("notify_on_every_fall", True))
            notify_sms = bool(s.get("notify_sms", False))
            notify_phone = bool(s.get("notify_phone", False))
            if s.get("active_dataset_code"):
                dataset_code = str(s.get("active_dataset_code") or dataset_code)
            if s.get("active_op_code"):
                op_code = str(s.get("active_op_code") or op_code).upper()
            if s.get("active_model_code"):
                model_code = str(s.get("active_model_code") or model_code)
    elif _table_exists(conn, "settings"):
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM settings WHERE resident_id=%s LIMIT 1", (int(resident_id),))
            s = cur.fetchone() or {}
        if isinstance(s, dict):
            notify_on_every_fall = bool(s.get("notify_on_every_fall", True))
            notify_sms = bool(s.get("notify_sms", False))
            notify_phone = bool(s.get("notify_phone", False))
            if s.get("active_model_code"):
                model_code = str(s.get("active_model_code") or model_code)

    if _table_exists(conn, "caregivers"):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, email, phone FROM caregivers WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
                (int(resident_id),),
            )
            cg = cur.fetchone() or {}
        if isinstance(cg, dict):
            caregiver_name = str(cg.get("name") or "").strip()
            caregiver_email = str(cg.get("email") or "").strip()
            caregiver_phone = str(cg.get("phone") or "").strip()

    if not notify_on_every_fall:
        return

    get_notification_manager().handle_event(
        SafeGuardEvent(
            event_id=str(event_id),
            resident_id=int(resident_id),
            location=location,
            probability=float(p_fall),
            uncertainty=0.0,
            threshold=0.5,
            margin=max(0.0, float(p_fall) - 0.5),
            triage_state="fall",
            safe_alert=True,
            recall_alert=True,
            model_code=str(model_code),
            dataset_code=str(dataset_code),
            op_code=str(op_code),
            source="events.test_fall",
            meta={"event_db_id": int(event_id)},
        ),
        NotificationPreferences(
            phone_enabled=bool(notify_phone),
            sms_enabled=bool(notify_sms),
            email_enabled=True,
            caregiver_name=caregiver_name,
            caregiver_phone=caregiver_phone,
            caregiver_email=caregiver_email,
        ),
    )


@router.get("/api/events")
@router.get("/api/v1/events")
def list_events(
    resident_id: Optional[int] = None,
    page: int = 1,
    page_size: int = 50,
    start_date: Optional[str] = None,  # YYYY-MM-DD (local UI date)
    end_date: Optional[str] = None,    # YYYY-MM-DD (local UI date, inclusive)
    event_type: Optional[str] = None,  # exact type (e.g., "fall", "uncertain"), or None/"All"
    status: Optional[str] = None,      # pending_review/confirmed_fall/false_alarm/dismissed, or None/"All"
    model: Optional[str] = None,       # GCN/TCN, or None/"All"
    limit: Optional[int] = None,       # legacy: /api/events?limit=500
) -> Dict[str, Any]:
    """List events with server-side pagination (+ optional filters).

    Backward compatible:
      - If the caller uses `limit` (old client), we treat it as `page_size` for page 1.
    """
    # Basic guardrails
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 1
    page = max(1, page)

    # page_size: prefer explicit page_size; but accept legacy `limit` for page 1
    try:
        page_size = int(page_size)
    except (TypeError, ValueError):
        page_size = 50

    if limit is not None and page == 1 and page_size == 50:
        # Old clients pass limit=500; keep behaviour similar but cap at 200.
        try:
            page_size = int(limit)
        except (TypeError, ValueError):
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

    # Convert date-only strings to datetime bounds (inclusive end_date by using < next_day)
    start_dt = None
    end_excl_dt = None
    try:
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
    except (TypeError, ValueError):
        start_dt = None
    try:
        if end_date:
            end_excl_dt = datetime.fromisoformat(end_date) + timedelta(days=1)
    except (TypeError, ValueError):
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
                    except (TypeError, json.JSONDecodeError):
                        meta = {"raw": meta}

                status_value = None
                if isinstance(meta, dict):
                    status_raw = meta.get("status")
                    if status_raw is not None:
                        status_value = str(status_raw).strip().lower() or None

                ts = r.get("ts")
                out.append(
                    {
                        "id": r.get("id"),
                        "event_time": ts,
                        "ts": ts,
                        "type": r.get("type"),
                        "severity": r.get("severity"),
                        "model_code": r.get("model_code"),
                        "status": status_value or "pending_review",
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
@router.get("/api/v1/events/summary")
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
        has_status = _has_col(conn, "events", "status")

        since = datetime.utcnow() - timedelta(hours=24)
        pending_24h = 0
        false_alarms_24h = 0
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

                if has_status:
                    cur.execute(
                        f"SELECT COUNT(*) AS n FROM events WHERE resident_id=%s AND LOWER(`status`) IN ('unreviewed','pending_review') AND `{time_col}` >= %s",
                        (rid, since),
                    )
                    pending_24h = int((cur.fetchone() or {}).get("n") or 0)

                    cur.execute(
                        f"SELECT COUNT(*) AS n FROM events WHERE resident_id=%s AND LOWER(`status`) IN ('false_alarm','false_positive') AND `{time_col}` >= %s",
                        (rid, since),
                    )
                    false_alarms_24h = int((cur.fetchone() or {}).get("n") or 0)
            except (MySQLError, RuntimeError, TypeError, ValueError) as e:
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
        }
    )


@router.put("/api/events/{event_id}/status")
@router.put("/api/v1/events/{event_id}/status")
def update_event_status(event_id: int, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Update event status for UI review flow."""
    status = str((payload or {}).get("status") or "").strip().lower()
    if not status:
        raise HTTPException(status_code=400, detail="Missing status")

    with get_conn_optional() as conn:
        if conn is None:
            return {
                "ok": True,
                "persisted": False,
                "reason": "db_unavailable",
                "event_id": int(event_id),
                "status": status,
            }

        variants = _detect_variants(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM events WHERE id=%s LIMIT 1", (int(event_id),))
            row = cur.fetchone()
            if not row:
                return {
                    "ok": True,
                    "persisted": False,
                    "reason": "event_not_found",
                    "event_id": int(event_id),
                    "status": status,
                }

            if variants.get("events") == "v2" and _has_col(conn, "events", "status"):
                cur.execute("UPDATE events SET status=%s WHERE id=%s", (status, int(event_id)))
            elif _has_col(conn, "events", "meta"):
                cur.execute("SELECT meta FROM events WHERE id=%s LIMIT 1", (int(event_id),))
                meta_row = cur.fetchone() or {}
                meta = meta_row.get("meta")
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except (TypeError, json.JSONDecodeError):
                        meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                meta["status"] = status
                cur.execute("UPDATE events SET meta=%s WHERE id=%s", (json.dumps(meta), int(event_id)))
            else:
                raise HTTPException(status_code=409, detail="Event status update unsupported for current schema")
        conn.commit()

    return {"ok": True, "persisted": True, "event_id": int(event_id), "status": status}


@router.post("/api/events/test_fall")
@router.post("/api/v1/events/test_fall")
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
                dispatch = dispatch_fall_notifications(
                    conn,
                    resident_id=int(rid),
                    event_id=int(new_id),
                    p_fall=0.99,
                    source="events.test_fall",
                )
                try:
                    _dispatch_safe_guard_from_event(
                        conn,
                        resident_id=int(rid),
                        event_id=int(new_id),
                        model_code="TCN",
                    )
                except Exception:
                    pass
                cur.execute("SELECT * FROM events WHERE id=%s", (new_id,))
                row = cur.fetchone()
                conn.commit()
                return {"ok": True, "event": _jsonable(row), "notification_dispatch": dispatch}

            meta = {"source": "ui_test"}
            cur.execute(
                """INSERT INTO events (resident_id, ts, type, severity, model_code, score, meta)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (rid, now, "fall", "high", "TCN", 0.99, json.dumps(meta)),
            )
            new_id = cur.lastrowid
            dispatch = dispatch_fall_notifications(
                conn,
                resident_id=int(rid),
                event_id=int(new_id),
                p_fall=0.99,
                source="events.test_fall",
            )
            try:
                _dispatch_safe_guard_from_event(
                    conn,
                    resident_id=int(rid),
                    event_id=int(new_id),
                    model_code="TCN",
                )
            except Exception:
                pass
            cur.execute("SELECT * FROM events WHERE id=%s", (new_id,))
            row = cur.fetchone()
            conn.commit()
            return {"ok": True, "event": _jsonable(row), "notification_dispatch": dispatch}


@router.post("/api/events/{event_id}/skeleton_clip")
@router.post("/api/v1/events/{event_id}/skeleton_clip")
def upload_skeleton_clip(event_id: int, payload: SkeletonClipPayload = Body(...)) -> Dict[str, Any]:
    """Persist a short skeleton-only clip and attach it to an existing event."""
    rid = int(payload.resident_id or 1)

    with get_conn_optional() as conn:
        if conn is None:
            raise HTTPException(status_code=503, detail="DB not available")
        store_event_clips, anonymize = _read_clip_privacy_flags(conn, rid)
        if not store_event_clips:
            return {"ok": True, "skipped": True, "reason": "store_event_clips_disabled"}

        with conn.cursor() as cur:
            cur.execute("SELECT id, resident_id, meta FROM events WHERE id=%s LIMIT 1", (int(event_id),))
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
        if int(row.get("resident_id") or 0) != rid:
            raise HTTPException(status_code=403, detail="Event does not belong to resident")

        t_ms = np.asarray(payload.t_ms, dtype=np.float32)
        xy = np.asarray(payload.xy, dtype=np.float32)
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise HTTPException(status_code=400, detail="xy must be shaped [T,J,2]")
        T = int(xy.shape[0])
        if t_ms.shape[0] != T:
            m = int(min(t_ms.shape[0], T))
            t_ms = t_ms[:m]
            xy = xy[:m]
            T = m

        if payload.conf is None:
            conf = np.ones((T, xy.shape[1]), dtype=np.float32)
        else:
            conf = np.asarray(payload.conf, dtype=np.float32)
            if conf.ndim != 2:
                raise HTTPException(status_code=400, detail="conf must be shaped [T,J]")
            if conf.shape[0] != T:
                m = int(min(conf.shape[0], T))
                conf = conf[:m]
                t_ms = t_ms[:m]
                xy = xy[:m]
                T = m

        if anonymize:
            xy = _anonymize_xy_inplace(xy)
            xy = np.round(xy, 4)
            conf = np.round(conf, 4)

        clips_dir = _event_clips_dir()
        fname = f"event_{int(event_id)}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.npz"
        fpath = clips_dir / fname
        try:
            np.savez_compressed(fpath, t_ms=t_ms, xy=xy, conf=conf)
        except (OSError, ValueError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to save clip: {e}")

        clip_rel = f"server/event_clips/{fname}"

        meta = row.get("meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (TypeError, json.JSONDecodeError):
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
            "anonymized": bool(anonymize),
            "saved_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        }

        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE events SET meta=%s WHERE id=%s", (json.dumps(meta), int(event_id)))
            conn.commit()
        except (MySQLError, RuntimeError, TypeError, ValueError):
            pass

        return {
            "ok": True,
            "event_id": int(event_id),
            "clip_path": clip_rel,
            "n_frames": int(T),
            "anonymized": bool(anonymize),
        }
