from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from applications.backend.routes import notifications as notifications_route


class _FakeStore:
    def list_recent_events(self, resident_id: int, limit: int = 50):
        return [
            {
                "id": "evt-1",
                "resident_id": int(resident_id),
                "ts": "2026-04-09T12:00:00+00:00",
                "channel": "telegram",
                "status": "sent",
                "message": "Safe Guard test",
                "event_id": "evt-1",
            }
        ][: int(limit)]


class _FakeManager:
    enabled = True

    def __init__(self):
        self.store = _FakeStore()


def _client():
    app = FastAPI()
    app.include_router(notifications_route.router)
    return TestClient(app)


def test_list_notifications_reads_safe_guard_store(monkeypatch):
    monkeypatch.setattr(notifications_route, "get_notification_manager", lambda: _FakeManager())
    client = _client()

    resp = client.get("/api/notifications?resident_id=7&limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["resident_id"] == 7
    assert body["db_available"] is True
    assert body["source"] == "safe_guard_sqlite"
    assert body["rows"][0]["status"] == "sent"
    assert body["rows"][0]["channel"] == "telegram"
