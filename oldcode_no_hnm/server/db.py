# server/db.py
import os
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor

# Read from env or fall back to defaults from create_db.sql
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "elder_fall_monitor")
DB_USER = os.getenv("DB_USER", "fall_app")
DB_PASSWORD = os.getenv("DB_PASSWORD", "strong_password_here")


@contextmanager
def get_conn():
    """
    Simple context manager for MySQL connections.

    Usage:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
    """
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        cursorclass=DictCursor,
        autocommit=True,
    )
    try:
        yield conn
    finally:
        conn.close()
