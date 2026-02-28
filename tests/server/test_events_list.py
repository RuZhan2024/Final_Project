from contextlib import contextmanager
from datetime import datetime

from fastapi.testclient import TestClient

from server.main import app
from server.routes import events as events_route


class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self._current = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self._conn.executed.append((sql, params))
        if not self._conn.responses:
            self._current = None
            return
        self._current = self._conn.responses.pop(0)

    def fetchone(self):
        if isinstance(self._current, dict):
            return self._current
        return {}

    def fetchall(self):
        if isinstance(self._current, list):
            return self._current
        return []


class _FakeConn:
    def __init__(self, responses):
        self.responses = list(responses)
        self.executed = []

    def cursor(self):
        return _FakeCursor(self)


def _with_conn(conn):
    @contextmanager
    def _cm():
        yield conn

    return _cm()


def test_list_events_pagination_contract_v1(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"n": 12},
            [
                {
                    "id": 101,
                    "ts": "2026-02-26T10:00:00",
                    "type": "fall",
                    "severity": "high",
                    "model_code": "GCN",
                    "score": 0.92,
                    "meta": "{\"source\":\"test\"}",
                }
            ],
        ]
    )

    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _with_conn(fake))
    monkeypatch.setattr(events_route, "_resident_exists", lambda _conn, _rid: True)
    monkeypatch.setattr(events_route, "_one_resident_id", lambda _conn: 1)
    monkeypatch.setattr(events_route, "_detect_variants", lambda _conn: {"settings": "v1", "events": "v1", "ops": "v1"})
    monkeypatch.setattr(events_route, "_event_time_col", lambda _conn: "ts")
    monkeypatch.setattr(events_route, "_event_prob_col", lambda _conn: "score")

    client = TestClient(app)
    resp = client.get("/api/events?resident_id=1&page=2&page_size=5")
    assert resp.status_code == 200

    body = resp.json()
    assert body["page"] == 2
    assert body["page_size"] == 5
    assert body["total"] == 12
    assert len(body["events"]) == 1
    assert body["events"][0]["p_fall"] == 0.92

    assert len(fake.executed) == 2
    _sql2, params2 = fake.executed[1]
    assert params2[-2:] == (5, 5)


def test_list_events_filter_contract_v2(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"n": 1},
            [
                {
                    "id": 201,
                    "ts": "2026-02-26T11:00:00",
                    "type": "fall",
                    "status": "pending_review",
                    "score": 0.87,
                    "operating_point_id": 3,
                    "model_code": "GCN",
                    "model_family": "GCN",
                    "notes": "n1",
                    "fa24h_snapshot": None,
                    "payload_json": None,
                }
            ],
        ]
    )

    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _with_conn(fake))
    monkeypatch.setattr(events_route, "_resident_exists", lambda _conn, _rid: True)
    monkeypatch.setattr(events_route, "_one_resident_id", lambda _conn: 1)
    monkeypatch.setattr(events_route, "_detect_variants", lambda _conn: {"settings": "v2", "events": "v2", "ops": "v2"})
    monkeypatch.setattr(events_route, "_event_time_col", lambda _conn: "event_time")
    monkeypatch.setattr(events_route, "_event_prob_col", lambda _conn: "p_fall")

    client = TestClient(app)
    resp = client.get(
        "/api/events"
        "?resident_id=1"
        "&start_date=2026-02-01"
        "&end_date=2026-02-27"
        "&event_type=Fall"
        "&status=pending_review"
        "&model=GCN"
        "&page=1&page_size=10"
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["total"] == 1
    assert body["events"][0]["status"] == "pending_review"
    assert body["events"][0]["model_code"] == "GCN"

    _sql1, params1 = fake.executed[0]
    assert params1[0] == 1
    assert isinstance(params1[1], datetime)
    assert isinstance(params1[2], datetime)
    assert params1[3] == "fall"
    assert params1[4] == "pending_review"
    assert params1[5] == "GCN"
    assert params1[6] == "GCN"
