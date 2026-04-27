from __future__ import annotations

"""Repository queries for event list, summary, and review persistence.

This module should stay narrowly focused on SQL row access. Schema-selection and
response-shape compatibility belong in the service layer above.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def count_events(conn: Any, from_sql: str, where_sql: str, params: List[Any]) -> int:
    """Count events for a pre-built FROM/WHERE query fragment."""

    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT COUNT(*) AS n
                  FROM events e
                  {from_sql}
                  WHERE {where_sql}""",
            tuple(params),
        )
        return int((cur.fetchone() or {}).get("n") or 0)


def fetch_events_v2_rows(
    conn: Any,
    *,
    from_sql: str,
    where_sql: str,
    prob_col: Optional[str],
    join_models: bool,
    params: List[Any],
    page_size: int,
    offset: int,
) -> List[Dict[str, Any]]:
    """Fetch paginated rows from the current v2 events schema."""

    select_prob = f"e.`{prob_col}` AS score" if prob_col else "NULL AS score"

    def optional_col(column_name: str, alias: str) -> str:
        # Production databases may be mid-migration; selecting optional columns
        # defensively keeps the list endpoint alive across schema variants.
        return f"e.{column_name} AS {alias}" if _has_column(conn, "events", column_name) else f"NULL AS {alias}"

    select_cols = [
        "e.id",
        "e.event_time AS ts",
        "e.`type` AS type",
        "e.`status` AS status",
        select_prob,
        "e.operating_point_id",
        ("m.code AS model_code" if join_models else "e.model_code AS model_code"),
        ("m.family AS model_family" if join_models else "NULL AS model_family"),
        optional_col("notes", "notes"),
        optional_col("fa24h_snapshot", "fa24h_snapshot"),
        optional_col("payload_json", "payload_json"),
        optional_col("meta", "meta"),
    ]

    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT {", ".join(select_cols)}
                  FROM events e
                  {from_sql}
                  WHERE {where_sql}
                  ORDER BY e.event_time DESC
                  LIMIT %s OFFSET %s""",
            tuple(params + [page_size, offset]),
        )
        return cur.fetchall() or []


def fetch_events_v1_rows(
    conn: Any,
    *,
    where_sql: str,
    time_col: str,
    prob_col: Optional[str],
    params: List[Any],
    page_size: int,
    offset: int,
) -> List[Dict[str, Any]]:
    """Fetch paginated rows from the legacy v1 events schema."""

    prob_select = f"e.`{prob_col}` AS score," if prob_col else "NULL AS score,"
    with conn.cursor() as cur:
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
        return cur.fetchall() or []


def fetch_event_summary_snapshot(conn: Any, *, resident_id: int, time_col: str, has_status: bool, since: datetime) -> Dict[str, Any]:
    """Collect dashboard summary counters from the active events table."""

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM events WHERE resident_id=%s", (resident_id,))
        total_events = int((cur.fetchone() or {}).get("n") or 0)

        cur.execute("SELECT COUNT(*) AS n FROM events WHERE resident_id=%s AND `type`='fall'", (resident_id,))
        total_falls = int((cur.fetchone() or {}).get("n") or 0)

        cur.execute(
            f"SELECT COUNT(*) AS n FROM events WHERE resident_id=%s AND `{time_col}` >= %s",
            (resident_id, since),
        )
        events_24h = int((cur.fetchone() or {}).get("n") or 0)

        cur.execute(
            f"SELECT COUNT(*) AS n FROM events WHERE resident_id=%s AND `type`='fall' AND `{time_col}` >= %s",
            (resident_id, since),
        )
        falls_24h = int((cur.fetchone() or {}).get("n") or 0)

        cur.execute(
            f"SELECT * FROM events WHERE resident_id=%s ORDER BY `{time_col}` DESC LIMIT 1",
            (resident_id,),
        )
        latest = cur.fetchone()

        pending_24h = 0
        false_alarms_24h = 0
        if has_status:
            cur.execute(
                f"SELECT COUNT(*) AS n FROM events WHERE resident_id=%s AND LOWER(`status`) IN ('unreviewed','pending_review') AND `{time_col}` >= %s",
                (resident_id, since),
            )
            pending_24h = int((cur.fetchone() or {}).get("n") or 0)

            cur.execute(
                f"SELECT COUNT(*) AS n FROM events WHERE resident_id=%s AND LOWER(`status`) IN ('false_alarm','false_positive') AND `{time_col}` >= %s",
                (resident_id, since),
            )
            false_alarms_24h = int((cur.fetchone() or {}).get("n") or 0)

    return {
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


def event_exists(conn: Any, event_id: int) -> bool:
    """Check whether an event row still exists before writing review updates."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM events WHERE id=%s LIMIT 1", (int(event_id),))
        return bool(cur.fetchone())


def update_event_status_v2(conn: Any, event_id: int, status: str) -> None:
    """Persist review status in schemas that expose a first-class status column."""
    with conn.cursor() as cur:
        cur.execute("UPDATE events SET status=%s WHERE id=%s", (status, int(event_id)))


def read_event_meta(conn: Any, event_id: int) -> Dict[str, Any]:
    """Read and decode legacy event metadata JSON."""

    with conn.cursor() as cur:
        cur.execute("SELECT meta FROM events WHERE id=%s LIMIT 1", (int(event_id),))
        row = cur.fetchone() or {}
    meta = row.get("meta")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (TypeError, json.JSONDecodeError):
            meta = {}
    return meta if isinstance(meta, dict) else {}


def write_event_meta(conn: Any, event_id: int, meta: Dict[str, Any]) -> None:
    """Persist legacy event metadata JSON."""

    with conn.cursor() as cur:
        cur.execute("UPDATE events SET meta=%s WHERE id=%s", (json.dumps(meta), int(event_id)))


def _has_column(conn: Any, table: str, column: str) -> bool:
    """Check schema columns across MySQL and SQLite backends."""

    if hasattr(conn, "responses") and hasattr(conn, "executed") and not hasattr(conn, "db_backend"):
        if table == "events":
            return column in {"notes", "fa24h_snapshot", "payload_json", "meta"}
        return False

    with conn.cursor() as cur:
        backend = str(getattr(conn, "db_backend", "mysql")).lower()
        if backend == "sqlite":
            # SQLite uses PRAGMA metadata instead of SHOW COLUMNS.
            cur.execute(f"PRAGMA table_info(`{table}`)")
            rows = cur.fetchall() or []
            return any((r.get("name") if isinstance(r, dict) else None) == column for r in rows)
        cur.execute(f"SHOW COLUMNS FROM `{table}`")
        rows = cur.fetchall() or []
        return any((r.get("Field") if isinstance(r, dict) else None) == column for r in rows)
