from fastapi.testclient import TestClient

from applications.backend.application import create_app
from applications.backend.config import get_app_config


def test_health_endpoint_returns_ok():
    client = TestClient(create_app(include_runtime_routes=False))
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert isinstance(body.get("ts"), str)


def test_ready_endpoint_returns_sqlite_readiness(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "ready.sqlite3"))
    get_app_config.cache_clear()

    client = TestClient(create_app(include_runtime_routes=False))
    resp = client.get("/api/ready")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["db_backend"] == "sqlite"
    assert "sqlite_path=" in body["detail"]

    get_app_config.cache_clear()
