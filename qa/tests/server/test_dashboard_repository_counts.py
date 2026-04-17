from __future__ import annotations

from applications.backend.repositories.dashboard_repository import load_today_counts


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.current = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        self.current = self.conn.responses.pop(0) if self.conn.responses else None

    def fetchone(self):
        if isinstance(self.current, dict):
            return self.current
        return {}


class _FakeConn:
    def __init__(self, responses=None, db_backend="sqlite"):
        self.responses = list(responses or [])
        self.executed = []
        self.db_backend = db_backend

    def cursor(self):
        return _FakeCursor(self)


def test_load_today_counts_prefers_current_events_type_schema():
    fake = _FakeConn(
        responses=[
            {"c": 4},
            {"c": 1},
        ],
        db_backend="sqlite",
    )

    counts = load_today_counts(
        fake,
        resident_id=3,
        table_exists=lambda _c, table: table == "events",
        col_exists=lambda _c, table, col: (table, col) in {
            ("events", "type"),
            ("events", "resident_id"),
            ("events", "event_time"),
        },
    )

    assert counts == {"falls_detected": 4, "false_alarms": 1, "confirmed_falls": 0}
    assert "UPPER(type)" in fake.executed[0][0]
    assert "DATE(event_time)" in fake.executed[0][0]
    assert fake.executed[0][1] == (3,)
    assert fake.executed[1][1] == (3,)


def test_load_today_counts_includes_confirmed_falls_when_status_exists():
    fake = _FakeConn(
        responses=[
            {"c": 4},
            {"c": 1},
            {"c": 2},
        ],
        db_backend="sqlite",
    )

    counts = load_today_counts(
        fake,
        resident_id=3,
        table_exists=lambda _c, table: table == "events",
        col_exists=lambda _c, table, col: (table, col) in {
            ("events", "type"),
            ("events", "status"),
            ("events", "resident_id"),
            ("events", "event_time"),
        },
    )

    assert counts == {"falls_detected": 4, "false_alarms": 1, "confirmed_falls": 2}
    assert "LOWER(status)='confirmed_fall'" in fake.executed[2][0]
    assert fake.executed[2][1] == (3,)
