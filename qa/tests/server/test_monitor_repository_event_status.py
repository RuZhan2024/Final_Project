from __future__ import annotations

from applications.backend.repositories.monitor_repository import insert_monitor_event


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.current = None
        self.lastrowid = conn.lastrowid

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        self.current = self.conn.responses.pop(0) if self.conn.responses else None
        self.lastrowid = self.conn.lastrowid

    def fetchall(self):
        if isinstance(self.current, list):
            return self.current
        return []


class _FakeConn:
    def __init__(self, responses=None, lastrowid=11, db_backend="sqlite"):
        self.responses = list(responses or [])
        self.lastrowid = lastrowid
        self.executed = []
        self.db_backend = db_backend

    def cursor(self):
        return _FakeCursor(self)


def test_insert_monitor_event_writes_pending_review_when_status_column_exists():
    fake = _FakeConn(
        responses=[
            [{"name": "id"}, {"name": "resident_id"}, {"name": "status"}, {"name": "meta"}],
            None,
        ],
        db_backend="sqlite",
    )

    event_id = insert_monitor_event(
        fake,
        resident_id=1,
        event_type="fall",
        severity="high",
        model_code="TCN",
        score=0.93,
        meta={"source": "monitor"},
        table_exists=lambda _c, table: table == "events",
    )

    assert event_id == 11
    insert_sql, insert_params = fake.executed[-1]
    assert "status" in insert_sql
    assert "pending_review" in insert_params


def test_insert_monitor_event_omits_status_when_column_missing():
    fake = _FakeConn(
        responses=[
            [{"name": "id"}, {"name": "resident_id"}, {"name": "meta"}],
            None,
        ],
        db_backend="sqlite",
    )

    insert_monitor_event(
        fake,
        resident_id=1,
        event_type="fall",
        severity="high",
        model_code="TCN",
        score=0.93,
        meta={"source": "monitor"},
        table_exists=lambda _c, table: table == "events",
    )

    insert_sql, insert_params = fake.executed[-1]
    assert "status" not in insert_sql
    assert "pending_review" not in insert_params
