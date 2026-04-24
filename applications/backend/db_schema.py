from __future__ import annotations

"""Schema-discovery and lightweight migration helpers for backend tables."""

from typing import Any, Dict, List, Optional, Set

try:
    from pymysql.err import MySQLError  # type: ignore
except (ImportError, ModuleNotFoundError):
    class MySQLError(Exception):
        pass


_TABLE_CACHE: Optional[Set[str]] = None
_COL_CACHE: Dict[str, Set[str]] = {}


def list_tables(conn: Any) -> Set[str]:
    """List tables once per process and cache the result for repeated schema checks."""
    global _TABLE_CACHE
    if _TABLE_CACHE is not None:
        return _TABLE_CACHE
    try:
        with conn.cursor() as cur:
            backend = str(getattr(conn, "db_backend", "mysql")).lower()
            if backend == "sqlite":
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                rows = cur.fetchall() or []
            else:
                cur.execute("SHOW TABLES")
                rows = cur.fetchall() or []
        tables: Set[str] = set()
        for row in rows:
            if isinstance(row, dict):
                value = row.get("name")
                if value is None:
                    value = next(iter(row.values()), None)
                if value:
                    tables.add(str(value))
            else:
                try:
                    tables.add(str(row[0]))
                except (TypeError, IndexError, KeyError):
                    pass
        _TABLE_CACHE = tables
        return tables
    except (MySQLError, RuntimeError, AttributeError, TypeError, ValueError):
        _TABLE_CACHE = set()
        return set()


def cols(conn: Any, table: str) -> Set[str]:
    """Return the cached set of column names for one table."""
    if table in _COL_CACHE:
        return _COL_CACHE[table]
    try:
        with conn.cursor() as cur:
            backend = str(getattr(conn, "db_backend", "mysql")).lower()
            if backend == "sqlite":
                cur.execute(f"PRAGMA table_info(`{table}`)")
                rows = cur.fetchall() or []
            else:
                cur.execute(f"SHOW COLUMNS FROM `{table}`")
                rows = cur.fetchall() or []
    except (MySQLError, RuntimeError, AttributeError, TypeError, ValueError):
        _COL_CACHE[table] = set()
        return set()
    names = {
        row.get("Field") or row.get("name")
        for row in rows
        if isinstance(row, dict) and (row.get("Field") or row.get("name"))
    }
    _COL_CACHE[table] = set(names)
    return _COL_CACHE[table]


def has_col(conn: Any, table: str, col: str) -> bool:
    return col in cols(conn, table)


def table_exists(conn: Any, table_name: str) -> bool:
    return table_name in list_tables(conn)


def col_exists(conn: Any, table_name: str, col_name: str) -> bool:
    return col_name in cols(conn, table_name)


def ensure_system_settings_schema(conn: Any) -> None:
    """Add missing settings columns needed by the current backend/frontend contract."""
    try:
        if not table_exists(conn, "system_settings"):
            return
        backend = str(getattr(conn, "db_backend", "mysql")).lower()

        wanted: Dict[str, str] = {
            "fall_threshold": "DECIMAL(6,4) NULL",
            "store_event_clips": "TINYINT(1) NOT NULL DEFAULT 0",
            "anonymize_skeleton_data": "TINYINT(1) NOT NULL DEFAULT 1",
            "active_dataset_code": "VARCHAR(16) NOT NULL DEFAULT 'caucafall'",
            "active_op_code": "VARCHAR(8) NOT NULL DEFAULT 'OP-2'",
            "mc_enabled": "TINYINT(1) NOT NULL DEFAULT 0",
            "mc_M": "INT NOT NULL DEFAULT 10",
            "mc_M_confirm": "INT NOT NULL DEFAULT 25",
            "notify_sms": "TINYINT(1) NOT NULL DEFAULT 0",
            "notify_phone": "TINYINT(1) NOT NULL DEFAULT 0",
            "fps": "INT NOT NULL DEFAULT 30",
            "window_size": "INT NOT NULL DEFAULT 48",
            "stride": "INT NOT NULL DEFAULT 12",
        }

        alters: List[str] = []
        for col, ddl in wanted.items():
            if not col_exists(conn, "system_settings", col):
                alters.append(f"ADD COLUMN `{col}` {ddl}")

        if alters:
            # SQLite cannot batch ADD COLUMN statements, so apply them one by one there.
            with conn.cursor() as cur:
                if backend == "sqlite":
                    for alter in alters:
                        cur.execute(f"ALTER TABLE `system_settings` {alter}")
                else:
                    cur.execute(f"ALTER TABLE `system_settings` {', '.join(alters)}")
            conn.commit()
            _COL_CACHE.pop("system_settings", None)
    except (MySQLError, RuntimeError, AttributeError, TypeError, ValueError):
        return


def ensure_caregivers_table(conn: Any) -> None:
    """Create the caregivers table or add the Telegram column for older schemas."""
    try:
        with conn.cursor() as cur:
            backend = str(getattr(conn, "db_backend", "mysql")).lower()
            if not table_exists(conn, "caregivers"):
                if backend == "sqlite":
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS caregivers (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          resident_id INTEGER NOT NULL,
                          name TEXT NULL,
                          email TEXT NULL,
                          phone TEXT NULL,
                          telegram_chat_id TEXT NULL,
                          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                          updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                else:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS caregivers (
                          id INT AUTO_INCREMENT PRIMARY KEY,
                          resident_id INT NOT NULL,
                          name VARCHAR(120) NULL,
                          email VARCHAR(200) NULL,
                          phone VARCHAR(80) NULL,
                          telegram_chat_id VARCHAR(120) NULL,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                          INDEX idx_resident (resident_id)
                        )
                        """
                    )
            elif not col_exists(conn, "caregivers", "telegram_chat_id"):
                if backend == "sqlite":
                    cur.execute("ALTER TABLE caregivers ADD COLUMN telegram_chat_id TEXT NULL")
                else:
                    cur.execute("ALTER TABLE caregivers ADD COLUMN `telegram_chat_id` VARCHAR(120) NULL")
        conn.commit()
        global _TABLE_CACHE
        _TABLE_CACHE = None
        _COL_CACHE.pop("caregivers", None)
    except (MySQLError, RuntimeError, AttributeError, TypeError, ValueError):
        return
