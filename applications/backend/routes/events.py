from __future__ import annotations

import json

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass

from ..core import _anonymize_xy_inplace, _event_clips_dir, _read_clip_privacy_flags
from ..db import get_conn_optional
from ..db_schema import col_exists, cols, has_col, table_exists
from ..deploy_ops import detect_variants
from ..event_schema import event_prob_col, event_time_col
from ..json_utils import jsonable as _jsonable
from ..notifications import get_notification_manager
from ..repositories.residents_repository import one_resident_id, resident_exists
from ..schemas import SkeletonClipPayload
from ..services.events_service import (
    EventsDeps,
    build_event_skeleton_clip_response,
    build_events_list_response,
    build_events_summary_response,
    create_test_fall_event,
    dispatch_safe_guard_from_event,
    persist_event_skeleton_clip,
    persist_event_status,
)
from ..services.value_coercion import coerce_bool


router = APIRouter()

# Keep a minimal monkeypatch surface for legacy route-level tests while the
# canonical implementation lives behind service/repository boundaries.
_detect_variants = detect_variants
_has_col = has_col
_one_resident_id = one_resident_id
_dispatch_safe_guard_from_event = dispatch_safe_guard_from_event


def _events_deps() -> EventsDeps:
    return EventsDeps(
        resident_exists=resident_exists,
        one_resident_id=_one_resident_id,
        detect_variants=_detect_variants,
        event_time_col=event_time_col,
        event_prob_col=event_prob_col,
        cols=cols,
        has_col=_has_col,
        table_exists=table_exists,
        jsonable=_jsonable,
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
        return create_test_fall_event(
            conn,
            deps=_events_deps(),
            dispatch_safe_guard=_dispatch_safe_guard_from_event,
            get_notification_manager=get_notification_manager,
            coerce_bool=coerce_bool,
        )


@router.post("/api/events/{event_id}/skeleton_clip")
@router.post("/api/v1/events/{event_id}/skeleton_clip")
def upload_skeleton_clip(event_id: int, payload: SkeletonClipPayload = Body(...)) -> Dict[str, Any]:
    """Persist a short skeleton-only clip and attach it to an existing event."""
    rid = int(payload.resident_id or 1)

    with get_conn_optional() as conn:
        if conn is None:
            raise HTTPException(status_code=503, detail="DB not available")
        try:
            return persist_event_skeleton_clip(
                conn,
                event_id=int(event_id),
                resident_id=rid,
                payload=payload,
                read_clip_privacy_flags=_read_clip_privacy_flags,
                anonymize_xy_inplace=_anonymize_xy_inplace,
                event_clips_dir=_event_clips_dir,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=str(exc))


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
