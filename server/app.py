# server/app.py
"""FastAPI entrypoint.

Keep this module tiny so `uvicorn server.app:app` remains stable.
The actual app assembly lives in :mod:`server.main`.
"""

from __future__ import annotations

from .main import app
