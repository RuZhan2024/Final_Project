from __future__ import annotations

from typing import Any, Optional

from .db_schema import cols


def event_time_col(conn: Any) -> str:
    event_cols = cols(conn, "events")
    for candidate in ("ts", "event_time", "created_at"):
        if candidate in event_cols:
            return candidate
    return "event_time"


def event_prob_col(conn: Any) -> Optional[str]:
    event_cols = cols(conn, "events")
    for candidate in ("score", "p_fall"):
        if candidate in event_cols:
            return candidate
    return None
