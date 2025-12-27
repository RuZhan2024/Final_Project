#!/usr/bin/env python3
"""
Simple PyMySQL connection helper.

NOTE: You asked to hardcode credentials. This is okay for local dev,
but avoid committing real passwords to Git.
"""
from __future__ import annotations

import contextlib
import pymysql

# Hardcoded local dev settings
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "fall_app"
DB_PASS = "strong_password_here"
DB_NAME = "elder_fall_monitor"

@contextlib.contextmanager
def get_conn():
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
        charset="utf8mb4",
    )
    try:
        yield conn
    finally:
        conn.close()
