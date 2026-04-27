from __future__ import annotations

"""Small coercion helpers shared by backend service layers."""

from typing import Any


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Coerce mixed DB/YAML/API values into a stable boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        if not normalized:
            return bool(default)
    # Unknown non-empty strings fall back to the caller default rather than truthy string rules.
    return bool(default if value is None else value)
