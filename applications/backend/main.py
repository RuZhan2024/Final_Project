from __future__ import annotations

"""ASGI module exporting the default backend application object."""

from .config import get_app_config
from .application import create_app


def _compute_allowed_origins() -> list[str]:
    """Legacy compatibility helper for tests and older callers."""
    get_app_config.cache_clear()
    return list(get_app_config().cors_allowed_origins)


app = create_app()
