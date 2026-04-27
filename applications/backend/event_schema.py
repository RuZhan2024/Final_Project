from __future__ import annotations

"""Helpers for resolving schema-dependent event column names."""

from typing import Any, Optional

from .db_schema import cols


def event_time_col(conn: Any) -> str:
    """Return the timestamp column supported by the current events table schema."""
    event_cols = cols(conn, "events")
    for candidate in ("ts", "event_time", "created_at"):
        if candidate in event_cols:
            return candidate
    return "event_time"


def event_prob_col(conn: Any) -> Optional[str]:
    """Return the probability column supported by the current events table schema."""
    event_cols = cols(conn, "events")
    for candidate in ("score", "p_fall"):
        if candidate in event_cols:
            return candidate
    return None
