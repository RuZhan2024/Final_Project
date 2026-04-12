from __future__ import annotations

import os

from datetime import datetime, timezone
from typing import Any, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


def get_app_timezone():
    tz_name = str(os.getenv("APP_TIMEZONE", "Europe/London")).strip() or "Europe/London"
    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    return datetime.now().astimezone().tzinfo or timezone.utc


def ensure_utc_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def serialize_event_timestamp(value: Any) -> Any:
    dt = ensure_utc_datetime(value)
    if dt is None:
        return value
    return dt.isoformat()


def format_local_event_timestamp(value: Any) -> str:
    dt = ensure_utc_datetime(value)
    if dt is None:
        return str(value)
    local_dt = dt.astimezone(get_app_timezone())
    return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
