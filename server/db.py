# server/db.py
from __future__ import annotations

import contextlib
from typing import Generator, Optional

import pymysql
from pymysql.cursors import DictCursor

# Hard-coded DB config (edit as needed)
DB_HOST: str = "127.0.0.1"
DB_PORT: int = 3306
DB_USER: str = "fall_app"
DB_PASS: str = "strong_password_here"
DB_NAME: str = "elder_fall_monitor"


@contextlib.contextmanager
def get_conn() -> Generator[pymysql.connections.Connection, None, None]:
    """Yield a PyMySQL connection (DictCursor, autocommit)."""
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
    )
    try:
        yield conn
    finally:
        conn.close()
