# applications/backend/app.py
"""Stable ASGI entrypoint.

Keep this module tiny so deployment commands such as
`uvicorn applications.backend.app:app`
remain stable while app assembly lives in :mod:`server.application`.
"""

from __future__ import annotations

from .application import create_app

app = create_app()
