from fastapi.testclient import TestClient

from applications.backend.main import app
from applications.backend.routes import settings as settings_route


def test_settings_falls_back_when_db_unavailable(monkeypatch):
    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(settings_route, "get_conn", _boom)

    client = TestClient(app)
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["db_available"] is False
    assert "system" in body
    assert "deploy" in body


def test_events_summary_includes_today_when_db_unavailable():
    client = TestClient(app)
    resp = client.get("/api/events/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "today" in body
    assert set(body["today"].keys()) >= {"falls", "pending", "false_alarms"}


def test_apply_yaml_override_tolerates_exceptions(monkeypatch):
    monkeypatch.setattr(
        settings_route,
        "_derive_ops_params_from_yaml",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    system = {}
    settings_route._apply_yaml_override(system)
    assert system.get("active_op_code") == "OP-2"
