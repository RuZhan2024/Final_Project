#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Database connection helpers for MySQL and SQLite backends."""

from __future__ import annotations

import os
import sqlite3
import sys

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from .config import get_app_config
try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass


def get_db_backend() -> str:
    return get_app_config().db_backend


def _require_env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(
            f"Missing required env var {name}. "
            "Set DB_HOST/DB_PORT/DB_USER/DB_PASS/DB_NAME or disable DB endpoints."
        )
    return str(v)


def _sqlite_path() -> Path:
    return get_app_config().sqlite_path


def _sqlite_placeholder_sql(sql: str) -> str:
    out = []
    i = 0
    while i < len(sql):
        if sql[i:i + 2] == "%s":
            out.append("?")
            i += 2
            continue
        out.append(sql[i])
        i += 1
    return "".join(out)


class SQLiteCursorWrapper:
    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._cursor.close()
        return False

    def execute(self, sql: str, params: Any = None):
        sql = _sqlite_placeholder_sql(sql)
        if params is None:
            self._cursor.execute(sql)
        else:
            self._cursor.execute(sql, params)
        return self

    def executemany(self, sql: str, seq):
        self._cursor.executemany(_sqlite_placeholder_sql(sql), seq)
        return self

    def fetchone(self):
        row = self._cursor.fetchone()
        return dict(row) if row is not None else None

    def fetchall(self):
        return [dict(r) for r in (self._cursor.fetchall() or [])]

    @property
    def lastrowid(self):
        return self._cursor.lastrowid


class SQLiteConnectionWrapper:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self.db_backend = "sqlite"

    def cursor(self):
        return SQLiteCursorWrapper(self._conn.cursor())

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def execute(self, sql: str, params: Any = None):
        cur = self._conn.cursor()
        try:
            if params is None:
                cur.execute(_sqlite_placeholder_sql(sql))
            else:
                cur.execute(_sqlite_placeholder_sql(sql), params)
            self._conn.commit()
        finally:
            cur.close()


def _ensure_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS residents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS caregivers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resident_id INTEGER NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            telegram_chat_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL UNIQUE,
            family TEXT,
            name TEXT NOT NULL,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS operating_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            code TEXT NOT NULL,
            name TEXT NOT NULL,
            thr_detect REAL,
            thr_low_conf REAL,
            thr_high_conf REAL,
            est_fa24h REAL,
            est_recall REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS system_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resident_id INTEGER NOT NULL,
            monitoring_enabled INTEGER NOT NULL DEFAULT 0,
            api_online INTEGER NOT NULL DEFAULT 1,
            last_latency_ms INTEGER,
            active_model_code TEXT NOT NULL DEFAULT 'TCN',
            active_operating_point INTEGER,
            alert_cooldown_sec INTEGER NOT NULL DEFAULT 3,
            notify_on_every_fall INTEGER NOT NULL DEFAULT 1,
            fall_threshold REAL DEFAULT 0.7100,
            store_event_clips INTEGER NOT NULL DEFAULT 0,
            anonymize_skeleton_data INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            active_dataset_code TEXT NOT NULL DEFAULT 'caucafall',
            active_op_code TEXT NOT NULL DEFAULT 'OP-2',
            mc_enabled INTEGER NOT NULL DEFAULT 0,
            mc_M INTEGER NOT NULL DEFAULT 10,
            mc_M_confirm INTEGER NOT NULL DEFAULT 25,
            notify_sms INTEGER NOT NULL DEFAULT 0,
            notify_phone INTEGER NOT NULL DEFAULT 0,
            fps INTEGER NOT NULL DEFAULT 30,
            window_size INTEGER NOT NULL DEFAULT 48,
            stride INTEGER NOT NULL DEFAULT 12
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resident_id INTEGER NOT NULL,
            ts TEXT DEFAULT CURRENT_TIMESTAMP,
            event_time TEXT DEFAULT CURRENT_TIMESTAMP,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            type TEXT,
            severity TEXT,
            model_code TEXT,
            operating_point_id INTEGER,
            score REAL,
            p_fall REAL,
            p_uncertain REAL,
            p_nonfall REAL,
            status TEXT,
            alert_sent INTEGER DEFAULT 0,
            notes TEXT,
            meta TEXT,
            payload_json TEXT
        );
        CREATE TABLE IF NOT EXISTS notifications_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resident_id INTEGER NOT NULL,
            ts TEXT DEFAULT CURRENT_TIMESTAMP,
            channel TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT,
            event_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS heartbeat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latency_ms INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    row = conn.execute("SELECT COUNT(*) FROM residents").fetchone()
    if row and int(row[0] or 0) == 0:
        conn.execute("INSERT INTO residents (name) VALUES ('Demo Resident')")

    row = conn.execute("SELECT COUNT(*) FROM caregivers").fetchone()
    if row and int(row[0] or 0) == 0:
        conn.execute(
            "INSERT INTO caregivers (resident_id, name, email, phone, telegram_chat_id) VALUES (1, 'Demo Caregiver', 'caregiver@example.com', '0000000000', '')"
        )

    row = conn.execute("SELECT COUNT(*) FROM models").fetchone()
    if row and int(row[0] or 0) == 0:
        conn.executemany(
            "INSERT INTO models (code, family, name, description) VALUES (?, ?, ?, ?)",
            [
                ("TCN", "TCN", "TCN", "Temporal Convolution Network"),
                ("GCN", "GCN", "GCN", "Graph Convolution Network"),
                ("HYBRID", "HYBRID", "Hybrid", "GCN + TCN (hybrid)"),
            ],
        )

    row = conn.execute("SELECT COUNT(*) FROM operating_points").fetchone()
    if row and int(row[0] or 0) == 0:
        conn.executemany(
            "INSERT INTO operating_points (model_id, code, name, thr_detect, thr_low_conf, thr_high_conf) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "OP-1", "High Sensitivity", 0.20, 0.1560, 0.20),
                (1, "OP-2", "Balanced", 0.71, 0.5538, 0.71),
                (1, "OP-3", "Low Sensitivity", 0.95, 0.7410, 0.95),
            ],
        )

    row = conn.execute("SELECT COUNT(*) FROM system_settings").fetchone()
    if row and int(row[0] or 0) == 0:
        conn.execute(
            "INSERT INTO system_settings (resident_id, active_model_code, active_operating_point, alert_cooldown_sec, notify_on_every_fall) VALUES (1, 'TCN', 2, 3, 1)"
        )

    conn.commit()


@contextmanager
def get_conn() -> Iterator[object]:
    if get_db_backend() == "sqlite":
        path = _sqlite_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = sqlite3.connect(str(path), timeout=10.0)
        raw.row_factory = sqlite3.Row
        _ensure_sqlite_schema(raw)
        conn = SQLiteConnectionWrapper(raw)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
        return

    try:
        import pymysql  # type: ignore
        from pymysql.cursors import DictCursor  # type: ignore
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            "PyMySQL is not installed. Install it with: pip install pymysql"
        ) from e

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
    allow_test_mysql = (
        "pymysql" in sys.modules
        or os.getenv("FALL_DETECTION_ALLOW_TEST_DB", "").strip() in {"1", "true", "yes"}
    )
    if (
        os.getenv("PYTEST_CURRENT_TEST")
        and not allow_test_mysql
        and get_db_backend() != "sqlite"
    ):
        yield None
        return

    conn_cm = None
    try:
        conn_cm = get_conn()
        conn = conn_cm.__enter__()
    except (MySQLError, OSError, RuntimeError, ValueError, TypeError, sqlite3.Error):
        yield None
        return
    try:
        yield conn
    finally:
        if conn_cm is not None:
            conn_cm.__exit__(None, None, None)
