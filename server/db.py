#!/usr/bin/env python3
"""server/db.py

MySQL helper used by the FastAPI server.

This module supports two modes:
1) **Local dev (default)**: uses hard-coded defaults (so the server can start
   without exporting env vars).
2) **Env override**: set DB_HOST/DB_PORT/DB_USER/DB_PASS/DB_NAME to override.

Notes
-----
- If you see:
    RuntimeError: 'cryptography' package is required for sha256_password or caching_sha2_password
  it means your MySQL user is using an auth plugin that needs the `cryptography`
  Python package when connecting via PyMySQL. Fix by installing:

      pip install cryptography

  …or by changing the MySQL user's auth plugin to `mysql_native_password`.

- For MySQL users created as '...@localhost', connecting with host='localhost'
  is often safer than '127.0.0.1' because MySQL treats those hosts differently.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator

import pymysql
from pymysql.cursors import DictCursor


# ---------------------------------------------------------------------------
# Default DB config (hard-coded as requested)
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    # Prefer 'localhost' (can match MySQL user '...@localhost')
    "DB_HOST": "localhost",
    "DB_PORT": 3306,
    "DB_USER": "fall_app",
    "DB_PASS": "strong_password_here",
    "DB_NAME": "elder_fall_monitor",
}


def db_config() -> Dict[str, Any]:
    """Return effective DB config (env overrides hard-coded defaults)."""
    host = os.getenv("DB_HOST", str(_DEFAULTS["DB_HOST"]))
    port_raw = os.getenv("DB_PORT", str(_DEFAULTS["DB_PORT"]))
    try:
        port = int(port_raw)
    except Exception:
        port = int(_DEFAULTS["DB_PORT"])

    return {
        "host": host,
        "port": port,
        "user": os.getenv("DB_USER", str(_DEFAULTS["DB_USER"])),
        "password": os.getenv("DB_PASS", str(_DEFAULTS["DB_PASS"])),
        "database": os.getenv("DB_NAME", str(_DEFAULTS["DB_NAME"])),
    }


@contextmanager
def get_conn() -> Iterator[pymysql.connections.Connection]:
    """Yield a PyMySQL connection (DictCursor, autocommit).

    Raises a readable RuntimeError if the connection fails.
    """
    cfg = db_config()

    try:
        conn = pymysql.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            cursorclass=DictCursor,
            autocommit=True,
            charset="utf8mb4",
            connect_timeout=5,
            read_timeout=10,
            write_timeout=10,
        )
    except RuntimeError as e:
        # PyMySQL raises RuntimeError for missing cryptography when using
        # caching_sha2_password / sha256_password.
        msg = str(e)
        if "cryptography" in msg and "caching_sha2_password" in msg:
            raise RuntimeError(
                "MySQL auth requires the 'cryptography' package. "
                "Install it with: pip install cryptography (and add it to requirements)."
            ) from e
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to MySQL at {cfg['host']}:{cfg['port']} as {cfg['user']} to DB '{cfg['database']}'. "
            f"Original error: {e}"
        ) from e

    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass
