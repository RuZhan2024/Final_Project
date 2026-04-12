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
    _anonymize_xy_inplace,
    _col_exists,
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
from ..schemas import SkeletonClipPayload
from ..services.events_service import (
    EventsDeps,
    build_event_skeleton_clip_response,
    build_events_list_response,
    build_events_summary_response,
    persist_event_status,
)
from ..services.value_coercion import coerce_bool


router = APIRouter()


def _events_deps() -> EventsDeps:
    return EventsDeps(
        resident_exists=_resident_exists,
        one_resident_id=_one_resident_id,
        detect_variants=_detect_variants,
        event_time_col=_event_time_col,
        event_prob_col=_event_prob_col,
        has_col=_has_col,
        table_exists=_table_exists,
        jsonable=_jsonable,
    )


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
) -> Dict[str, Any]:
    notify_on_every_fall = True
    notify_sms = False
    notify_phone = False
    caregiver_name = ""
    caregiver_email = ""
    caregiver_phone = ""
    caregiver_telegram_chat_id = ""

    if _table_exists(conn, "system_settings"):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM system_settings WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
                (int(resident_id),),
            )
            s = cur.fetchone() or {}
        if isinstance(s, dict):
            notify_on_every_fall = coerce_bool(s.get("notify_on_every_fall", True), True)
            notify_sms = coerce_bool(s.get("notify_sms", False), False)
            notify_phone = coerce_bool(s.get("notify_phone", False), False)
            if s.get("active_dataset_code"):
                dataset_code = str(s.get("active_dataset_code") or dataset_code)
            if s.get("active_op_code"):
                op_code = str(s.get("active_op_code") or op_code).upper()
            if s.get("active_model_code"):
                model_code = str(s.get("active_model_code") or model_code)

    if _table_exists(conn, "caregivers"):
        select_cols = "name, email, phone"
        if _col_exists(conn, "caregivers", "telegram_chat_id"):
            select_cols += ", telegram_chat_id"
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {select_cols} FROM caregivers WHERE resident_id=%s ORDER BY id ASC LIMIT 1",
                (int(resident_id),),
            )
            cg = cur.fetchone() or {}
        if isinstance(cg, dict):
            caregiver_name = str(cg.get("name") or "").strip()
            caregiver_email = str(cg.get("email") or "").strip()
            caregiver_phone = str(cg.get("phone") or "").strip()
            caregiver_telegram_chat_id = str(cg.get("telegram_chat_id") or "").strip()

    if not notify_on_every_fall:
        return {
            "enabled": False,
            "tier": "disabled",
            "reason": "notify_disabled",
            "actions": {"telegram": False},
            "enqueued": False,
            "state": "disabled",
            "audit_backend": "sqlite",
        }

    dispatch = get_notification_manager().handle_event(
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
            telegram_enabled=bool(notify_on_every_fall),
            caregiver_name=caregiver_name,
            caregiver_telegram_chat_id=caregiver_telegram_chat_id,
        ),
    )
    return {
        "enabled": bool(dispatch.enabled),
        "tier": dispatch.tier,
        "reason": dispatch.reason,
        "actions": dispatch.actions,
        "enqueued": bool(dispatch.enqueued),
        "state": dispatch.state,
        "audit_backend": dispatch.audit_backend,
    }


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
        return build_events_list_response(
            conn,
            resident_id=resident_id,
            page=page,
            page_size=page_size,
            start_date=start_date,
            end_date=end_date,
            event_type=event_type,
            status=status,
            model=model,
            limit=limit,
            deps=_events_deps(),
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
        try:
            return build_events_summary_response(conn, resident_id, _events_deps())
        except (MySQLError, RuntimeError, TypeError, ValueError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to compute events summary: {e}")


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
        try:
            return persist_event_status(conn, event_id, status, _events_deps())
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))


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
                    add("status", "pending_review")
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
                dispatch = {
                    "enabled": False,
                    "tier": "disabled",
                    "reason": "safe_guard_not_attempted",
                    "actions": {"telegram": False},
                    "enqueued": False,
                    "state": "disabled",
                    "audit_backend": "sqlite",
                }
                try:
                    dispatch = _dispatch_safe_guard_from_event(
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
            dispatch = {
                "enabled": False,
                "tier": "disabled",
                "reason": "safe_guard_not_attempted",
                "actions": {"telegram": False},
                "enqueued": False,
                "state": "disabled",
                "audit_backend": "sqlite",
            }
            try:
                dispatch = _dispatch_safe_guard_from_event(
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


@router.get("/api/events/{event_id}/skeleton_clip")
@router.get("/api/v1/events/{event_id}/skeleton_clip")
def get_skeleton_clip(event_id: int, resident_id: Optional[int] = None) -> Dict[str, Any]:
    rid = int(resident_id or 1)

    with get_conn_optional() as conn:
        if conn is None:
            raise HTTPException(status_code=503, detail="DB not available")
        try:
            return build_event_skeleton_clip_response(
                conn,
                event_id=int(event_id),
                resident_id=rid,
                event_clips_dir=_event_clips_dir(),
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
