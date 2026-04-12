from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..core import _jsonable
from ..time_utils import serialize_event_timestamp
from ..repositories.events_repository import (
    count_events,
    event_exists,
    fetch_event_summary_snapshot,
    fetch_events_v1_rows,
    fetch_events_v2_rows,
    read_event_meta,
    update_event_status_v2,
    write_event_meta,
)


@dataclass(frozen=True)
class EventsDeps:
    resident_exists: Callable[[Any, int], bool]
    one_resident_id: Callable[[Any], int]
    detect_variants: Callable[[Any], Dict[str, str]]
    event_time_col: Callable[[Any], str]
    event_prob_col: Callable[[Any], Optional[str]]
    has_col: Callable[[Any, str, str], bool]
    table_exists: Callable[[Any, str], bool]
    jsonable: Callable[[Any], Any] = _jsonable


def normalize_pagination(page: int, page_size: int, limit: Optional[int]) -> tuple[int, int, int]:
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 1
    page = max(1, page)

    try:
        page_size = int(page_size)
    except (TypeError, ValueError):
        page_size = 50

    if limit is not None and page == 1 and page_size == 50:
        try:
            page_size = int(limit)
        except (TypeError, ValueError):
            pass

    page_size = min(200, max(1, page_size))
    offset = (page - 1) * page_size
    return page, page_size, offset


def normalize_filter_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "all":
        return None
    return normalized


def parse_date_bounds(start_date: Optional[str], end_date: Optional[str]) -> tuple[Optional[datetime], Optional[datetime]]:
    start_dt = None
    end_exclusive = None
    try:
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
    except (TypeError, ValueError):
        start_dt = None
    try:
        if end_date:
            end_exclusive = datetime.fromisoformat(end_date) + timedelta(days=1)
    except (TypeError, ValueError):
        end_exclusive = None
    return start_dt, end_exclusive


def resolve_resident_id(conn: Any, resident_id: Optional[int], deps: EventsDeps) -> int:
    if resident_id and deps.resident_exists(conn, resident_id):
        return int(resident_id)
    return int(deps.one_resident_id(conn))


def build_events_list_response(
    conn: Any,
    *,
    resident_id: Optional[int],
    page: int,
    page_size: int,
    start_date: Optional[str],
    end_date: Optional[str],
    event_type: Optional[str],
    status: Optional[str],
    model: Optional[str],
    limit: Optional[int],
    deps: EventsDeps,
) -> Dict[str, Any]:
    page, page_size, offset = normalize_pagination(page, page_size, limit)
    start_date = normalize_filter_value(start_date)
    end_date = normalize_filter_value(end_date)
    event_type = normalize_filter_value(event_type)
    status = normalize_filter_value(status)
    model = normalize_filter_value(model)
    start_dt, end_excl_dt = parse_date_bounds(start_date, end_date)

    rid = resolve_resident_id(conn, resident_id, deps)
    variants = deps.detect_variants(conn)
    time_col = deps.event_time_col(conn)
    prob_col = deps.event_prob_col(conn)

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
        has_model_id = deps.has_col(conn, "events", "model_id")
        has_models_table = deps.table_exists(conn, "models")
        join_models = bool(has_model_id and has_models_table)
        from_sql = "LEFT JOIN models m ON m.id = e.model_id" if join_models else ""

        if status is not None:
            where.append("LOWER(e.`status`)=%s")
            params.append(status.lower())

        if model is not None:
            upper_model = model.upper()
            if join_models:
                where.append("(UPPER(m.code)=%s OR UPPER(m.family)=%s)")
                params.extend([upper_model, upper_model])
            else:
                where.append("UPPER(e.model_code)=%s")
                params.append(upper_model)

        where_sql = " AND ".join(where)
        total = count_events(conn, from_sql, where_sql, params)
        rows = fetch_events_v2_rows(
            conn,
            from_sql=from_sql,
            where_sql=where_sql,
            prob_col=prob_col,
            join_models=join_models,
            params=params,
            page_size=page_size,
            offset=offset,
        )
        events = [map_v2_event_row(row) for row in rows]
        return deps.jsonable(
            {
                "resident_id": rid,
                "events": events,
                "total": total,
                "page": page,
                "page_size": page_size,
            }
        )

    if model is not None:
        where.append("UPPER(e.model_code)=%s")
        params.append(model.upper())

    where_sql = " AND ".join(where)
    total = count_events(conn, "", where_sql, params)
    rows = fetch_events_v1_rows(
        conn,
        where_sql=where_sql,
        time_col=time_col,
        prob_col=prob_col,
        params=params,
        page_size=page_size,
        offset=offset,
    )
    events = [map_v1_event_row(row) for row in rows]
    return deps.jsonable(
        {
            "resident_id": rid,
            "events": events,
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


def build_events_summary_response(conn: Any, resident_id: Optional[int], deps: EventsDeps) -> Dict[str, Any]:
    rid = resolve_resident_id(conn, resident_id, deps)
    time_col = deps.event_time_col(conn)
    has_status = deps.has_col(conn, "events", "status")
    summary = fetch_event_summary_snapshot(
        conn,
        resident_id=rid,
        time_col=time_col,
        has_status=has_status,
        since=datetime.utcnow() - timedelta(hours=24),
    )
    return deps.jsonable({"resident_id": rid, **summary})


def persist_event_status(conn: Any, event_id: int, status: str, deps: EventsDeps) -> Dict[str, Any]:
    if not event_exists(conn, event_id):
        return {
            "ok": True,
            "persisted": False,
            "reason": "event_not_found",
            "event_id": int(event_id),
            "status": status,
        }

    variants = deps.detect_variants(conn)
    if variants.get("events") == "v2" and deps.has_col(conn, "events", "status"):
        update_event_status_v2(conn, event_id, status)
    elif deps.has_col(conn, "events", "meta"):
        meta = read_event_meta(conn, event_id)
        meta["status"] = status
        write_event_meta(conn, event_id, meta)
    else:
        raise ValueError("Event status update unsupported for current schema")

    conn.commit()
    return {"ok": True, "persisted": True, "event_id": int(event_id), "status": status}


def build_event_skeleton_clip_response(
    conn: Any,
    *,
    event_id: int,
    resident_id: int,
    event_clips_dir: Path,
) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, resident_id, meta FROM events WHERE id=%s LIMIT 1", (int(event_id),))
        row = cur.fetchone()

    if not row:
        raise LookupError(f"Event {event_id} not found")
    if int(row.get("resident_id") or 0) != int(resident_id):
        raise PermissionError("Event does not belong to resident")

    meta = parse_raw_meta(row.get("meta"))
    clip_meta = meta.get("skeleton_clip")
    if not isinstance(clip_meta, dict):
        raise FileNotFoundError("No skeleton clip stored for this event")

    rel_path = str(clip_meta.get("path") or "").strip()
    if not rel_path:
        raise FileNotFoundError("No skeleton clip stored for this event")

    clips_root = event_clips_dir.resolve()
    clip_path = (clips_root / Path(rel_path).name).resolve()
    try:
        clip_path.relative_to(clips_root)
    except ValueError as exc:
        raise FileNotFoundError("clip_not_found") from exc

    if not clip_path.exists() or not clip_path.is_file():
        raise FileNotFoundError("clip_not_found")

    try:
        with np.load(clip_path, allow_pickle=False) as data:
            t_ms = np.asarray(data["t_ms"], dtype=np.float32)
            xy = np.asarray(data["xy"], dtype=np.float32)
            conf = np.asarray(data["conf"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError(f"Failed to load skeleton clip: {exc}") from exc

    return {
        "ok": True,
        "event_id": int(event_id),
        "clip": {
            "path": rel_path,
            "n_frames": int(xy.shape[0]) if xy.ndim >= 1 else 0,
            "n_joints": int(xy.shape[1]) if xy.ndim >= 2 else 0,
            "pre_s": clip_meta.get("pre_s"),
            "post_s": clip_meta.get("post_s"),
            "dataset_code": clip_meta.get("dataset_code"),
            "mode": clip_meta.get("mode"),
            "op_code": clip_meta.get("op_code"),
            "anonymized": bool(clip_meta.get("anonymized", True)),
            "saved_at": clip_meta.get("saved_at"),
        },
        "t_ms": t_ms.tolist(),
        "xy": xy.tolist(),
        "conf": conf.tolist(),
    }


def map_v2_event_row(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = parse_meta_fields(row)
    source_raw = meta.get("event_source") or meta.get("input_source")
    source = str(source_raw).strip().lower() if source_raw is not None else "realtime"
    ts = serialize_event_timestamp(row.get("ts"))
    return {
        "id": row.get("id"),
        "event_time": ts,
        "ts": ts,
        "type": row.get("type"),
        "status": row.get("status"),
        "score": row.get("score"),
        "p_fall": row.get("score"),
        "model_code": row.get("model_code") or row.get("model_family"),
        "operating_point_id": row.get("operating_point_id"),
        "source": source or "realtime",
        "meta": meta,
    }


def map_v1_event_row(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = parse_raw_meta(row.get("meta"))
    status_raw = meta.get("status")
    source_raw = meta.get("event_source") or meta.get("input_source")
    ts = serialize_event_timestamp(row.get("ts"))
    return {
        "id": row.get("id"),
        "event_time": ts,
        "ts": ts,
        "type": row.get("type"),
        "severity": row.get("severity"),
        "model_code": row.get("model_code"),
        "status": (str(status_raw).strip().lower() if status_raw is not None else "pending_review"),
        "score": row.get("score"),
        "p_fall": row.get("score"),
        "source": (str(source_raw).strip().lower() if source_raw is not None else "realtime"),
        "meta": meta,
    }


def parse_meta_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = parse_raw_meta(row.get("meta"))
    for key in ("notes", "fa24h_snapshot", "payload_json"):
        if row.get(key) is not None:
            meta[key] = row.get(key)
    return meta


def parse_raw_meta(meta: Any) -> Dict[str, Any]:
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (TypeError, json.JSONDecodeError):
            meta = {"raw": meta}
    return meta if isinstance(meta, dict) else {}
