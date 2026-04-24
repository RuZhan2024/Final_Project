from __future__ import annotations

"""JSON-safe conversion helpers for backend responses and stored metadata."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


def jsonable(x: Any) -> Any:
    """Recursively convert backend values into JSON-serializable primitives."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, datetime):
        # Naive datetimes are treated as UTC to keep API/event payloads unambiguous.
        if x.tzinfo is None:
            return x.replace(tzinfo=timezone.utc).isoformat()
        return x.isoformat()
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except UnicodeDecodeError:
            return x.hex()
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    return str(x)
