from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from applications.backend.main import app
from applications.backend.routes import caregivers as caregivers_route
from applications.backend.routes import dashboard as dashboard_route
from applications.backend.routes import events as events_route
from applications.backend.routes import operating_points as ops_route
from applications.backend.routes import settings as settings_route
from applications.backend.routes import specs as specs_route


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


def test_settings_get_uses_current_repository_boundary(monkeypatch):
    fake = _FakeConn()

    def _load_snapshot(_conn, _resident_id, system, deploy):
        system.update(
            {
                "monitoring_enabled": True,
                "active_model_code": "TCN",
                "active_dataset_code": "caucafall",
                "active_op_code": "OP-2",
                "fall_threshold": 0.8,
            }
        )
        deploy.setdefault("window", {})["W"] = 48

    monkeypatch.setattr(settings_route, "get_conn", lambda: _cm_conn(fake))
    monkeypatch.setattr(settings_route, "load_settings_snapshot", _load_snapshot)
    monkeypatch.setattr(
        settings_route,
        "_derive_ops_params_from_yaml",
        lambda **_k: {"ui": {"op_code": "OP-2", "tau_low": 0.2, "tau_high": 0.8, "cooldown_s": 3}},
    )

    resp = TestClient(app).get("/api/settings?resident_id=1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["db_available"] is True
    assert body["system"]["active_model_code"] == "TCN"
    assert body["deploy"]["window"]["W"] == 48


def test_settings_update_uses_current_persist_boundary(monkeypatch):
    fake = _FakeConn()
    seen = {"resident_id": None, "threshold": None}

    def _persist(_conn, resident_id, payload):
        seen["resident_id"] = resident_id
        seen["threshold"] = payload.fall_threshold
        _conn.executed.append(("persist_settings_update", resident_id))
        return True

    monkeypatch.setattr(settings_route, "get_conn", lambda: _cm_conn(fake))
    monkeypatch.setattr(settings_route, "persist_settings_update", _persist)

    resp = TestClient(app).put("/api/settings?resident_id=7", json={"fall_threshold": 90})

    assert resp.status_code == 200
    assert resp.json()["persisted"] is True
    assert seen == {"resident_id": 7, "threshold": 0.9}


def test_caregivers_get_and_upsert_current_db_paths(monkeypatch):
    fake_get = _FakeConn(responses=[[{"id": 1, "resident_id": 1, "name": "Care A", "email": "a@x", "phone": "1"}]])
    monkeypatch.setattr(caregivers_route, "get_conn_optional", lambda: _cm_conn(fake_get))
    monkeypatch.setattr(caregivers_route, "ensure_caregivers_table", lambda _c: None)
    monkeypatch.setattr(caregivers_route, "table_exists", lambda _c, t: t == "caregivers")
    monkeypatch.setattr(caregivers_route, "col_exists", lambda _c, _t, _col: False)

    client = TestClient(app)
    r1 = client.get("/api/caregivers?resident_id=1")
    assert r1.status_code == 200
    assert r1.json()["caregivers"][0]["name"] == "Care A"

    fake_put = _FakeConn(
        responses=[
            {},  # existing caregiver lookup
            None,
            {"id": 2, "resident_id": 1, "name": "Care B", "email": "b@x", "phone": "2"},
        ],
        lastrowid=2,
    )
    monkeypatch.setattr(caregivers_route, "get_conn_optional", lambda: _cm_conn(fake_put))
    r2 = client.put("/api/caregivers", json={"resident_id": 1, "name": "Care B", "email": "b@x", "phone": "2"})

    assert r2.status_code == 200
    assert r2.json()["caregiver"]["name"] == "Care B"
    assert any("INSERT INTO caregivers" in sql for sql, _ in fake_put.executed)


def test_dashboard_summary_current_db_path(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"monitoring_enabled": 1, "active_model_code": "TCN", "active_model_id": None},
            {"c": 2},
            {"c": 1},
            {"c": 1},
            {"latency_ms": 77},
        ]
    )
    monkeypatch.setattr(dashboard_route, "get_conn", lambda: _cm_conn(fake))
    monkeypatch.setattr(dashboard_route, "resident_exists", lambda _c, _rid: True)
    monkeypatch.setattr(dashboard_route, "one_resident_id", lambda _c: 1)
    monkeypatch.setattr(dashboard_route, "table_exists", lambda _c, t: t in {"system_settings", "events", "heartbeat"})
    monkeypatch.setattr(
        dashboard_route,
        "col_exists",
        lambda _c, t, col: (t, col)
        in {
            ("events", "type"),
            ("events", "event_time"),
            ("events", "resident_id"),
            ("events", "status"),
            ("heartbeat", "latency_ms"),
        },
    )

    resp = TestClient(app).get("/api/dashboard/summary?resident_id=1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "alert"
    assert body["today"]["falls_detected"] == 2
    assert body["today"]["false_alarms"] == 1
    assert body["system"]["last_latency_ms"] == 77


def test_operating_points_v2_current_db_path(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"id": 9},
            [
                {
                    "id": 1,
                    "name": "Balanced",
                    "code": "OP-2",
                    "thr_detect": 0.5,
                    "thr_low_conf": 0.2,
                    "thr_high_conf": 0.8,
                    "est_fa24h": 0.4,
                    "est_recall": 0.9,
                }
            ],
        ]
    )
    monkeypatch.setattr(ops_route, "get_conn", lambda: _cm_conn(fake))
    monkeypatch.setattr(ops_route, "ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(ops_route, "detect_variants", lambda _c: {"settings": "v2", "events": "v2", "ops": "v2"})
    monkeypatch.setattr(ops_route, "table_exists", lambda _c, t: t == "models")

    resp = TestClient(app).get("/api/operating_points?model_code=CTR_GCN&dataset_code=caucafall")

    assert resp.status_code == 200
    body = resp.json()
    assert body["db_available"] is True
    assert body["operating_points"][0]["code"] == "OP-2"


def test_operating_points_v1_current_db_path(monkeypatch):
    fake = _FakeConn(
        responses=[
            [
                {
                    "id": 4,
                    "model_code": "TCN",
                    "name": "Balanced",
                    "threshold_low": 0.2,
                    "threshold_high": 0.8,
                    "cooldown_seconds": 3,
                    "code": "OP-2",
                }
            ]
        ]
    )
    monkeypatch.setattr(ops_route, "get_conn", lambda: _cm_conn(fake))
    monkeypatch.setattr(ops_route, "ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(ops_route, "detect_variants", lambda _c: {"settings": "v1", "events": "v1", "ops": "v1"})
    monkeypatch.setattr(ops_route, "table_exists", lambda _c, _t: False)

    resp = TestClient(app).get("/api/operating_points?model_code=TCN&dataset_code=caucafall")

    assert resp.status_code == 200
    assert resp.json()["operating_points"][0]["threshold_high"] == 0.8


def test_events_summary_current_db_path(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"n": 10},
            {"n": 4},
            {"n": 5},
            {"n": 3},
            {"id": 101, "type": "fall"},
            {"n": 2},
            {"n": 1},
        ]
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_resident_exists", lambda _c, _rid: True)
    monkeypatch.setattr(events_route, "_one_resident_id", lambda _c: 1)
    monkeypatch.setattr(events_route, "_event_time_col", lambda _c: "event_time")
    monkeypatch.setattr(events_route, "_has_col", lambda _c, t, col: (t, col) == ("events", "status"))

    resp = TestClient(app).get("/api/events/summary?resident_id=1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["db_available"] is True
    assert body["today"]["falls"] == 3
    assert body["today"]["pending"] == 2
    assert body["today"]["false_alarms"] == 1


def test_events_test_fall_v2_current_path(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"active_model_id": 2, "active_operating_point_id": 3},
            {"id": 7, "type": "fall"},
        ],
        lastrowid=7,
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_one_resident_id", lambda _c: 1)
    monkeypatch.setattr(events_route, "_detect_variants", lambda _c: {"settings": "v2", "events": "v2", "ops": "v2"})
    monkeypatch.setattr(events_route, "_has_col", lambda _c, t, col: (t, col) == ("system_settings", "active_model_id"))
    monkeypatch.setattr(
        events_route,
        "_cols_for_events",
        lambda _c, _t: {"resident_id", "model_id", "operating_point_id", "event_time", "type", "status", "p_fall", "notes", "payload_json"},
    )

    resp = TestClient(app).post("/api/events/test_fall")

    assert resp.status_code == 200
    inserts = [(sql, params) for sql, params in fake.executed if "INSERT INTO events" in sql]
    assert inserts
    assert "`status`" in inserts[0][0]
    assert "pending_review" in inserts[0][1]


def test_events_upload_skeleton_clip_success(monkeypatch, tmp_path: Path):
    fake = _FakeConn(
        responses=[
            {"id": 5, "resident_id": 1, "meta": "{}"},
            None,
        ]
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_read_clip_privacy_flags", lambda _c, _rid: (True, False))
    monkeypatch.setattr(events_route, "_event_clips_dir", lambda: tmp_path)

    resp = TestClient(app).post(
        "/api/events/5/skeleton_clip",
        json={
            "resident_id": 1,
            "t_ms": [0, 40],
            "xy": [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.1, 0.1], [1.1, 1.1]],
            ],
            "conf": [[1.0, 1.0], [1.0, 1.0]],
            "mode": "tcn",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    files = list(tmp_path.glob("event_5_*.npz"))
    assert files
    with np.load(files[0], allow_pickle=False) as z:
        assert np.asarray(z["xy"]).shape == (2, 2, 2)
    upd = [params for sql, params in fake.executed if "UPDATE events SET meta=%s WHERE id=%s" in sql]
    assert upd
    meta = json.loads(upd[0][0])
    assert meta["skeleton_clip"]["mode"] == "tcn"


def test_deploy_specs_endpoint_current_shape(monkeypatch):
    spec = SimpleNamespace(
        dataset="caucafall",
        arch="tcn",
        ckpt="/tmp/c.pt",
        temperature=1.107,
        ops={"OP-2": {"tau_low": 0.2, "tau_high": 0.85}},
        alert_cfg={"k": 2},
    )
    monkeypatch.setattr(specs_route, "_get_deploy_specs", lambda: {"caucafall_tcn": spec})

    resp = TestClient(app).get("/api/deploy/specs")

    assert resp.status_code == 200
    body = resp.json()
    assert body["datasets"] == ["caucafall"]
    assert body["specs"][0]["spec_key"] == "caucafall_tcn"
    assert body["specs"][0]["temperature"] == 1.107
