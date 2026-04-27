from __future__ import annotations

"""Repository queries backing dashboard summary cards.

These helpers answer a narrow set of dashboard questions: current system
settings, today's event counts, and the latest recorded latency. Anything about
fallbacks or response shaping belongs in the service layer.
"""

from typing import Any, Dict


def _today_filter_sql(conn: Any, column: str) -> str:
    """Return the backend-specific SQL fragment for local-date filtering."""
    if str(getattr(conn, "db_backend", "mysql")).lower() == "sqlite":
        return f"DATE({column}) = DATE('now', 'localtime')"
    return f"DATE({column})=CURDATE()"


def load_system_snapshot(
    conn: Any,
    resident_id: int,
    *,
    table_exists,
    col_exists,
) -> Dict[str, Any]:
    """Load the subset of system settings surfaced on the dashboard."""
    snapshot: Dict[str, Any] = {
        "model_name": "TCN",
        "monitoring_enabled": False,
    }

    if table_exists(conn, "system_settings"):
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM system_settings WHERE resident_id=%s ORDER BY id ASC LIMIT 1", (resident_id,))
            row = cur.fetchone() or {}
            if isinstance(row, dict):
                if "monitoring_enabled" in row:
                    snapshot["monitoring_enabled"] = bool(row.get("monitoring_enabled", 1))
                if row.get("active_model_code"):
                    snapshot["model_name"] = str(row.get("active_model_code") or snapshot["model_name"])

                active_model_id = row.get("active_model_id")
                if active_model_id and table_exists(conn, "models"):
                    # Prefer the human-friendly model name when the models table
                    # is present; otherwise fall back to the stored code.
                    cur.execute("SELECT * FROM models WHERE id=%s LIMIT 1", (active_model_id,))
                    model_row = cur.fetchone() or {}
                    if isinstance(model_row, dict):
                        snapshot["model_name"] = (
                            model_row.get("name")
                            or model_row.get("model_code")
                            or model_row.get("code")
                            or snapshot["model_name"]
                        )
        return snapshot

    return snapshot


def load_today_counts(
    conn: Any,
    resident_id: int,
    *,
    table_exists,
    col_exists,
) -> Dict[str, int]:
    """Load today's fall, false-alarm, and confirmed-fall counters."""
    falls = 0
    false_alarms = 0
    confirmed_falls = 0

    with conn.cursor() as cur:
        if table_exists(conn, "events"):
            type_col = None
            if col_exists(conn, "events", "type"):
                type_col = "type"
            elif col_exists(conn, "events", "event_type"):
                type_col = "event_type"

            if type_col is not None:
                # Support the repo-native events schema first, but keep compatibility
                # with older variants so dashboard counts do not silently drop to zero.
                time_col = "created_at"
                for candidate in ("event_time", "ts", "created_at"):
                    if col_exists(conn, "events", candidate):
                        time_col = candidate
                        break
                has_resident_id = col_exists(conn, "events", "resident_id")
                resident_filter = " AND resident_id=%s" if has_resident_id else ""
                params = (resident_id,) if has_resident_id else tuple()
                cur.execute(
                    "SELECT COUNT(*) AS c FROM events "
                    f"WHERE {_today_filter_sql(conn, time_col)} "
                    f"AND UPPER({type_col}) IN ('FALL','FALL_DETECTED','FALL_CONFIRMED')"
                    + resident_filter,
                    params,
                )
                row = cur.fetchone() or {}
                falls = int(row.get("c", 0)) if isinstance(row, dict) else int(list(row)[0])

                cur.execute(
                    "SELECT COUNT(*) AS c FROM events "
                    f"WHERE {_today_filter_sql(conn, time_col)} "
                    f"AND UPPER({type_col}) IN ('FALSE_ALARM','FALSE','FALSE_POSITIVE')"
                    + resident_filter,
                    params,
                )
                row = cur.fetchone() or {}
                false_alarms = int(row.get("c", 0)) if isinstance(row, dict) else int(list(row)[0])

                if col_exists(conn, "events", "status"):
                    cur.execute(
                        "SELECT COUNT(*) AS c FROM events "
                        f"WHERE {_today_filter_sql(conn, time_col)} "
                        "AND LOWER(status)='confirmed_fall'"
                        + resident_filter,
                        params,
                    )
                    row = cur.fetchone() or {}
                    confirmed_falls = int(row.get("c", 0)) if isinstance(row, dict) else int(list(row)[0])

                return {
                    "falls_detected": falls,
                    "false_alarms": false_alarms,
                    "confirmed_falls": confirmed_falls,
                }

        if table_exists(conn, "fall_events"):
            has_resident_id = col_exists(conn, "fall_events", "resident_id")
            resident_where = " AND resident_id=%s" if has_resident_id else ""
            params = (resident_id,) if has_resident_id else tuple()
            cur.execute(
                "SELECT "
                "SUM(CASE WHEN event_type='fall_detected' THEN 1 ELSE 0 END) AS falls_detected, "
                "SUM(CASE WHEN event_type='false_alarm' THEN 1 ELSE 0 END) AS false_alarms, "
                "SUM(CASE WHEN event_type IN ('confirmed_fall','fall_confirmed') THEN 1 ELSE 0 END) AS confirmed_falls "
                f"FROM fall_events WHERE {_today_filter_sql(conn, 'created_at')}" + resident_where,
                params,
            )
            row = cur.fetchone() or {}
            if isinstance(row, dict):
                falls = int(row.get("falls_detected") or 0)
                false_alarms = int(row.get("false_alarms") or 0)
                confirmed_falls = int(row.get("confirmed_falls") or 0)

    return {"falls_detected": falls, "false_alarms": false_alarms, "confirmed_falls": confirmed_falls}


def load_last_latency_ms(conn: Any, *, table_exists, col_exists) -> int | None:
    """Load the latest heartbeat latency when that table/column exists."""
    if not table_exists(conn, "heartbeat") or not col_exists(conn, "heartbeat", "latency_ms"):
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT latency_ms FROM heartbeat ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
    if isinstance(row, dict) and row.get("latency_ms") is not None:
        return int(row["latency_ms"])
    return None
