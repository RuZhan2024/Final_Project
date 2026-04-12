from contextlib import contextmanager

from applications.backend.routes import events as events_route


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

    def fetchone(self):
        if isinstance(self.current, dict):
            return self.current
        return {}

    def fetchall(self):
        if isinstance(self.current, list):
            return self.current
        return []


class _FakeConn:
    def __init__(self, responses=None, lastrowid=1):
        self.responses = list(responses or [])
        self.lastrowid = lastrowid
        self.executed = []
        self.commits = 0

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.commits += 1


def _cm_conn(conn):
    @contextmanager
    def _cm():
        yield conn

    return _cm()


def test_events_test_fall_v2_writes_pending_review(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"active_model_id": 2, "active_operating_point_id": 3},
            [
                {"Field": "model_id"},
                {"Field": "operating_point_id"},
                {"Field": "event_time"},
                {"Field": "type"},
                {"Field": "status"},
                {"Field": "p_fall"},
                {"Field": "notes"},
                {"Field": "payload_json"},
            ],
            {"id": 7, "type": "fall"},
        ],
        lastrowid=7,
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_one_resident_id", lambda _c: 1)
    monkeypatch.setattr(events_route, "_detect_variants", lambda _c: {"settings": "v2", "events": "v2", "ops": "v2"})
    monkeypatch.setattr(events_route, "_has_col", lambda _c, t, col: t == "system_settings" and col == "active_model_id")
    monkeypatch.setattr(
        events_route,
        "_dispatch_safe_guard_from_event",
        lambda *args, **kwargs: {
            "enabled": True,
            "tier": "tier1_high_confidence_fall",
            "reason": "test",
            "actions": {"telegram": True},
            "enqueued": True,
            "state": "enqueued",
            "audit_backend": "sqlite",
        },
    )

    body = events_route.test_fall()

    assert body["ok"] is True
    inserts = [(sql, params) for sql, params in fake.executed if "INSERT INTO events" in sql]
    assert inserts
    insert_sql, insert_params = inserts[0]
    assert "`status`" in insert_sql
    assert "pending_review" in insert_params
