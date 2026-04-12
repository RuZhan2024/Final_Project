from contextlib import contextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient

from applications.backend.main import app
from applications.backend import core as core_mod
from applications.backend.routes import caregivers as caregivers_route
from applications.backend.routes import dashboard as dashboard_route
from applications.backend.routes import events as events_route
from applications.backend.routes import monitor as monitor_route
from applications.backend.routes import operating_points as ops_route
from applications.backend.routes import settings as settings_route
from applications.backend.routes import specs as specs_route


def test_dashboard_summary_fallback_when_db_error(monkeypatch):
    @contextmanager
    def _broken():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr(dashboard_route, "get_conn", _broken)
    client = TestClient(app)
    resp = client.get("/api/dashboard/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["system"]["api_online"] is False
    assert "error" in body["system"]


def test_dashboard_summary_alias(monkeypatch):
    monkeypatch.setattr(dashboard_route, "dashboard_summary", lambda resident_id=None: {"ok": True, "resident_id": resident_id})
    client = TestClient(app)
    resp = client.get("/api/summary?resident_id=3")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "resident_id": 3}


def test_models_summary_uses_specs_and_db_fallback(monkeypatch):
    spec = SimpleNamespace(
        dataset="muvim",
        arch="gcn",
        ckpt="/tmp/mock.pt",
        data_cfg={"fps_default": 30},
        ops={"OP-2": {"tau_low": 0.2, "tau_high": 0.8}},
        alert_cfg={"k": 2},
    )
    monkeypatch.setattr(specs_route, "_get_deploy_specs", lambda: {"muvim_gcn": spec})

    @contextmanager
    def _broken():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr(specs_route, "get_conn", _broken)
    client = TestClient(app)
    resp = client.get("/api/models/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["models"]) == 1
    assert body["models"][0]["spec_key"] == "muvim_gcn"
    assert body["db_models"] == []


def test_deploy_specs_alias(monkeypatch):
    monkeypatch.setattr(specs_route, "deploy_specs", lambda: {"specs": [], "models": [], "datasets": []})
    client = TestClient(app)
    resp = client.get("/api/spec")
    assert resp.status_code == 200
    assert resp.json() == {"specs": [], "models": [], "datasets": []}


def test_replay_clips_routes(monkeypatch, tmp_path):
    clip_path = tmp_path / "demo.mp4"
    clip_path.write_bytes(b"demo-video")
    (tmp_path / "ignore.txt").write_text("ignore", encoding="utf-8")
    monkeypatch.setenv("REPLAY_CLIPS_DIR", str(tmp_path))

    client = TestClient(app)

    list_resp = client.get("/api/replay/clips")
    assert list_resp.status_code == 200
    body = list_resp.json()
    assert body["available"] is True
    assert len(body["clips"]) == 1
    assert body["clips"][0]["name"] == "demo.mp4"

    file_resp = client.get(body["clips"][0]["url"])
    assert file_resp.status_code == 200
    assert file_resp.content == b"demo-video"

    blocked_resp = client.get("/api/replay/clips/../secret.mp4")
    assert blocked_resp.status_code == 404


def test_operating_points_yaml_fallback(monkeypatch):
    @contextmanager
    def _broken():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr(ops_route, "get_conn", _broken)
    monkeypatch.setattr(
        ops_route,
        "_derive_ops_params_from_yaml",
        lambda dataset_code, model_code, op_code: {"ui": {"tau_low": 0.1, "tau_high": 0.9, "cooldown_s": 3.0}},
    )
    client = TestClient(app)
    resp = client.get("/api/operating_points?model_code=GCN&dataset_code=muvim")
    assert resp.status_code == 200
    body = resp.json()
    assert body["db_available"] is False
    assert len(body["operating_points"]) == 3


def test_caregivers_get_and_upsert_db_unavailable(monkeypatch):
    @contextmanager
    def _no_db():
        yield None

    monkeypatch.setattr(caregivers_route, "get_conn_optional", _no_db)
    client = TestClient(app)

    r1 = client.get("/api/caregivers?resident_id=7")
    assert r1.status_code == 200
    assert r1.json()["db_available"] is False

    r2 = client.put("/api/caregivers", json={"resident_id": 7, "name": "A"})
    assert r2.status_code == 503


def test_caregivers_table_missing_paths(monkeypatch):
    @contextmanager
    def _db():
        class _Conn:
            def cursor(self):
                class _Cur:
                    def __enter__(self):
                        return self

                    def __exit__(self, exc_type, exc, tb):
                        return False

                    def execute(self, *_a, **_k):
                        return None

                    def fetchall(self):
                        return []

                return _Cur()

        yield _Conn()

    monkeypatch.setattr(caregivers_route, "get_conn_optional", _db)
    monkeypatch.setattr(caregivers_route, "_ensure_caregivers_table", lambda _c: None)
    monkeypatch.setattr(caregivers_route, "_table_exists", lambda _c, _t: False)

    client = TestClient(app)
    r1 = client.get("/api/caregivers?resident_id=1")
    assert r1.status_code == 200
    assert r1.json()["caregivers"] == []

    r2 = client.put("/api/caregivers", json={"resident_id": 1, "name": "X"})
    assert r2.status_code == 500


def test_settings_update_fallback_inmem(monkeypatch):
    @contextmanager
    def _broken():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    calls = {"payload": None, "resident_id": None}

    def _capture(payload, resident_id):
        calls["payload"] = payload
        calls["resident_id"] = resident_id

    monkeypatch.setattr(settings_route, "get_conn", _broken)
    monkeypatch.setattr(settings_route, "apply_settings_update_inmem", _capture)

    client = TestClient(app)
    resp = client.put("/api/settings?resident_id=4", json={"fall_threshold": 85})
    assert resp.status_code == 200
    body = resp.json()
    assert body["persisted"] is False
    assert calls["resident_id"] == 4
    assert abs(float(calls["payload"].fall_threshold) - 0.85) < 1e-9


def test_events_skeleton_clip_validation_errors(monkeypatch):
    @contextmanager
    def _no_db():
        yield None

    monkeypatch.setattr(events_route, "get_conn_optional", _no_db)
    client = TestClient(app)
    resp = client.post("/api/events/1/skeleton_clip", json={"resident_id": 1, "t_ms": [1], "xy": [[[0, 0]]]})
    assert resp.status_code == 503


def test_monitor_reset_session(monkeypatch):
    core_mod._SESSION_STATE["to-reset"] = {"x": 1}
    client = TestClient(app)
    resp = client.post("/api/monitor/reset_session?session_id=to-reset")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert "to-reset" not in core_mod._SESSION_STATE
