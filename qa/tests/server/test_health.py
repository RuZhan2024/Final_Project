from fastapi.testclient import TestClient

from applications.backend.app import app


def test_health_endpoint_returns_ok():
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert isinstance(body.get("ts"), str)
