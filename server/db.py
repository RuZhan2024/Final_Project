#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""server/db.py

MySQL connector used by the FastAPI server.

This module is deliberately *import-safe*:
- If PyMySQL is not installed, the server can still start; DB endpoints will
  raise a clear error only when called.
- Missing DB environment variables also error at call time.

Env vars
--------
DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass


def _require_env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(
            f"Missing required env var {name}. "
            "Set DB_HOST/DB_PORT/DB_USER/DB_PASS/DB_NAME or disable DB endpoints."
        )
    return str(v)


@contextmanager
def get_conn() -> Iterator[object]:
    """Yield a PyMySQL connection (DictCursor).

    The caller is responsible for committing when needed.
    """
    try:
        import pymysql  # type: ignore
        from pymysql.cursors import DictCursor  # type: ignore
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            "PyMySQL is not installed. Install it with: pip install pymysql"
        ) from e

    # Allow env override, but fall back to defaults so local dev works
    # without exporting variables every time.
    host = _require_env("DB_HOST", "127.0.0.1")
    port = int(_require_env("DB_PORT", "3306"))
    user = _require_env("DB_USER", "fall_app")
    password = _require_env("DB_PASS", "strong_password_here")
    db = _require_env("DB_NAME", "elder_fall_monitor")

    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=db,
        cursorclass=DictCursor,
        autocommit=False,
    )
    try:
        yield conn
        conn.commit()
    finally:
        try:
            conn.close()
        except (MySQLError, OSError, RuntimeError):
            pass


@contextmanager
def get_conn_optional() -> Iterator[Optional[object]]:
    """Yield a connection when available, otherwise yield None.

    Use this in endpoints that should still work when:
    - PyMySQL isn't installed, or
    - the DB isn't reachable.
    """
    conn_cm = None
    try:
        conn_cm = get_conn()
        conn = conn_cm.__enter__()
    except (MySQLError, OSError, RuntimeError, ValueError, TypeError):
        # DB unavailable only during connection setup.
        yield None
        return
    try:
        yield conn
    finally:
        if conn_cm is not None:
            conn_cm.__exit__(None, None, None)
