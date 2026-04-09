from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from server.main import app
from server.routes import caregivers as caregivers_route
from server.routes import dashboard as dashboard_route
from server.routes import events as events_route
from server.routes import operating_points as ops_route
from server.routes import settings as settings_route
from server.routes import specs as specs_route


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


def test_settings_get_uses_settings_table(monkeypatch):
    fake = _FakeConn(
        responses=[
            {
                "monitoring_enabled": 1,
                "active_model_code": "GCN",
                "active_operating_point": 2,
                "active_op_code": "OP-3",
                "fall_threshold": 0.7,
                "alert_cooldown_sec": 4,
                "store_event_clips": 1,
                "anonymize_skeleton_data": 1,
                "require_confirmation": 0,
                "notify_on_every_fall": 1,
                "notify_sms": 1,
                "notify_phone": 0,
                "fps": 30,
                "window_size": 48,
                "stride": 12,
            }
        ]
    )
    monkeypatch.setattr(settings_route, "get_conn", lambda: _cm_conn(fake))
    monkeypatch.setattr(settings_route, "_ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(settings_route, "_table_exists", lambda _c, t: t == "settings")
    monkeypatch.setattr(settings_route, "_derive_ops_params_from_yaml", lambda **_k: {"ui": {"op_code": "OP-2", "tau_low": 0.1, "tau_high": 0.8, "cooldown_s": 3}})

    client = TestClient(app)
    resp = client.get("/api/settings?resident_id=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["db_available"] is True
    assert body["system"]["active_model_code"] == "GCN"
    assert body["deploy"]["window"]["W"] == 48


def test_settings_update_uses_settings_table(monkeypatch):
    fake = _FakeConn()
    monkeypatch.setattr(settings_route, "get_conn", lambda: _cm_conn(fake))
    monkeypatch.setattr(settings_route, "_table_exists", lambda _c, t: t == "settings")
    monkeypatch.setattr(settings_route, "_col_exists", lambda *_a, **_k: True)

    client = TestClient(app)
    resp = client.put(
        "/api/settings?resident_id=1",
        json={
            "monitoring_enabled": True,
            "active_model_code": "TCN",
            "alert_cooldown_sec": 5,
            "fall_threshold": 90,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["persisted"] is True
    assert any("UPDATE settings SET" in sql for sql, _ in fake.executed)


def test_caregivers_get_and_upsert_db_paths(monkeypatch):
    fake_get = _FakeConn(responses=[[{"id": 1, "resident_id": 1, "name": "Care A", "email": "a@x", "phone": "1"}]])
    monkeypatch.setattr(caregivers_route, "get_conn_optional", lambda: _cm_conn(fake_get))
    monkeypatch.setattr(caregivers_route, "_ensure_caregivers_table", lambda _c: None)
    monkeypatch.setattr(caregivers_route, "_table_exists", lambda _c, t: t == "caregivers")

    client = TestClient(app)
    r1 = client.get("/api/caregivers?resident_id=1")
    assert r1.status_code == 200
    assert len(r1.json()["caregivers"]) == 1

    fake_put = _FakeConn(
        responses=[
            {},  # SELECT id ... LIMIT 1 (no existing)
            None,  # INSERT ... (no fetch)
            {"id": 1, "resident_id": 1, "name": "Care B", "email": "b@x", "phone": "2"},
        ],
        lastrowid=1,
    )
    monkeypatch.setattr(caregivers_route, "get_conn_optional", lambda: _cm_conn(fake_put))
    r2 = client.put("/api/caregivers", json={"resident_id": 1, "name": "Care B", "email": "b@x", "phone": "2"})
    assert r2.status_code == 200
    assert r2.json()["caregiver"]["name"] == "Care B"


def test_dashboard_summary_db_path(monkeypatch):
    fake = _FakeConn(
        responses=[
            {
                "monitoring_enabled": 1,
                "active_model_code": "GCN",
                "active_model_id": None,
            },
            {"c": 2},  # today falls
            {"c": 1},  # today false
            {"latency_ms": 77},
        ]
    )
    monkeypatch.setattr(dashboard_route, "get_conn", lambda: _cm_conn(fake))
    monkeypatch.setattr(dashboard_route, "_resident_exists", lambda _c, _rid: True)
    monkeypatch.setattr(dashboard_route, "_one_resident_id", lambda _c: 1)

    def _table_exists(_c, t):
        return t in {"system_settings", "events", "heartbeat"}

    monkeypatch.setattr(dashboard_route, "_table_exists", _table_exists)
    monkeypatch.setattr(dashboard_route, "_col_exists", lambda _c, t, col: (t, col) in {("events", "event_type"), ("heartbeat", "latency_ms")})

    client = TestClient(app)
    resp = client.get("/api/dashboard/summary?resident_id=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "alert"
    assert body["today"]["falls_detected"] == 2
    assert body["system"]["last_latency_ms"] == 77


def test_dashboard_summary_scopes_event_counts_by_resident(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"monitoring_enabled": 1, "active_model_code": "GCN", "active_model_id": None},
            {"c": 3},  # falls today
            {"c": 1},  # false alarms today
        ]
    )
    monkeypatch.setattr(dashboard_route, "get_conn", lambda: _cm_conn(fake))
    monkeypatch.setattr(dashboard_route, "_resident_exists", lambda _c, _rid: True)
    monkeypatch.setattr(dashboard_route, "_one_resident_id", lambda _c: 1)
    monkeypatch.setattr(dashboard_route, "_table_exists", lambda _c, t: t in {"system_settings", "events"})
    monkeypatch.setattr(
        dashboard_route,
        "_col_exists",
        lambda _c, t, col: (t, col) in {("events", "event_type"), ("events", "resident_id")},
    )

    client = TestClient(app)
    resp = client.get("/api/dashboard/summary?resident_id=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["today"]["falls_detected"] == 3
    assert body["today"]["false_alarms"] == 1

    # Falls and false-alarm queries should both carry resident_id in params.
    event_count_queries = [q for q in fake.executed if "COUNT(*) AS c FROM events" in q[0]]
    assert len(event_count_queries) >= 2
    assert event_count_queries[0][1] == (5,)
    assert event_count_queries[1][1] == (5,)


def test_operating_points_v2_db_path(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"id": 9},  # model id
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
    monkeypatch.setattr(ops_route, "_ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(ops_route, "_detect_variants", lambda _c: {"settings": "v2", "events": "v2", "ops": "v2"})
    monkeypatch.setattr(ops_route, "_table_exists", lambda _c, t: t == "models")

    client = TestClient(app)
    resp = client.get("/api/operating_points?model_code=GCN&dataset_code=muvim")
    assert resp.status_code == 200
    body = resp.json()
    assert body["db_available"] is True
    assert body["operating_points"][0]["code"] == "OP-2"


def test_specs_models_summary_with_db_rows(monkeypatch):
    spec = SimpleNamespace(
        dataset="le2i",
        arch="tcn",
        ckpt="/tmp/x.pt",
        data_cfg={"fps_default": 25},
        ops={"OP-2": {"tau_low": 0.2, "tau_high": 0.85}},
        alert_cfg={"k": 2},
    )
    fake = _FakeConn(responses=[[{"id": 1, "code": "TCN", "name": "TCN"}]])
    monkeypatch.setattr(specs_route, "_get_deploy_specs", lambda: {"le2i_tcn": spec})
    monkeypatch.setattr(specs_route, "get_conn", lambda: _cm_conn(fake))

    client = TestClient(app)
    resp = client.get("/api/models/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["models"][0]["dataset_code"] == "le2i"
    assert len(body["db_models"]) == 1


def test_settings_update_v2_path(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"id": 2},  # active model lookup
            {"id": 1},  # select existing system_settings row
        ]
    )
    monkeypatch.setattr(settings_route, "get_conn", lambda: _cm_conn(fake))
    monkeypatch.setattr(settings_route, "_ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(settings_route, "_table_exists", lambda _c, t: t in {"system_settings", "models"})
    monkeypatch.setattr(settings_route, "_col_exists", lambda *_a, **_k: True)

    client = TestClient(app)
    resp = client.put(
        "/api/settings?resident_id=5",
        json={
            "monitoring_enabled": True,
            "fall_threshold": 0.8,
            "alert_cooldown_sec": 5,
            "store_event_clips": True,
            "anonymize_skeleton_data": False,
            "require_confirmation": True,
            "notify_on_every_fall": False,
            "active_dataset_code": "muvim",
            "active_op_code": "op-2",
            "mc_enabled": True,
            "mc_M": 10,
            "mc_M_confirm": 20,
            "active_model_code": "GCN",
            "active_operating_point": 3,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["persisted"] is True
    assert any("UPDATE system_settings SET" in sql for sql, _ in fake.executed)


def test_events_summary_db_path_with_status_counts(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"n": 10},  # total_events
            {"n": 4},   # total_falls
            {"n": 5},   # events_24h
            {"n": 3},   # falls_24h
            {"id": 101, "type": "fall"},  # latest
            {"n": 2},   # pending_24h
            {"n": 1},   # false_alarms_24h
        ]
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_resident_exists", lambda _c, _rid: True)
    monkeypatch.setattr(events_route, "_one_resident_id", lambda _c: 1)
    monkeypatch.setattr(events_route, "_event_time_col", lambda _c: "event_time")
    monkeypatch.setattr(events_route, "_cols", lambda _c, t: {"status"} if t == "events" else set())

    client = TestClient(app)
    resp = client.get("/api/events/summary?resident_id=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["db_available"] is True
    assert body["today"]["falls"] == 3
    assert body["today"]["pending"] == 2
    assert body["today"]["false_alarms"] == 1


def test_events_test_fall_v1_path(monkeypatch):
    fake = _FakeConn(responses=[{"id": 1, "resident_id": 1}], lastrowid=7)
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_one_resident_id", lambda _c: 1)
    monkeypatch.setattr(events_route, "_detect_variants", lambda _c: {"settings": "v1", "events": "v1", "ops": "v1"})

    client = TestClient(app)
    resp = client.post("/api/events/test_fall")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_deploy_specs_endpoint(monkeypatch):
    spec = SimpleNamespace(
        dataset="caucafall",
        arch="gcn",
        ckpt="/tmp/c.pt",
        ops={"OP-2": {"tau_low": 0.2, "tau_high": 0.85}},
        alert_cfg={"k": 2},
    )
    monkeypatch.setattr(specs_route, "_get_deploy_specs", lambda: {"caucafall_gcn": spec})
    client = TestClient(app)
    resp = client.get("/api/deploy/specs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["datasets"] == ["caucafall"]
    assert body["specs"][0]["spec_key"] == "caucafall_gcn"


def test_events_test_fall_v2_path(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"active_model_id": 2, "active_operating_point_id": 3},  # settings row
            [  # SHOW COLUMNS FROM events
                {"Field": "model_id"},
                {"Field": "operating_point_id"},
                {"Field": "event_time"},
                {"Field": "type"},
                {"Field": "status"},
                {"Field": "p_fall"},
                {"Field": "notes"},
                {"Field": "payload_json"},
            ],
            {"id": 7, "type": "fall"},  # selected inserted event
        ],
        lastrowid=7,
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_one_resident_id", lambda _c: 1)
    monkeypatch.setattr(events_route, "_detect_variants", lambda _c: {"settings": "v2", "events": "v2", "ops": "v2"})
    monkeypatch.setattr(events_route, "_has_col", lambda _c, t, col: t == "system_settings" and col == "active_model_id")

    client = TestClient(app)
    resp = client.post("/api/events/test_fall")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    inserts = [(sql, params) for sql, params in fake.executed if "INSERT INTO events" in sql]
    assert inserts
    insert_sql, insert_params = inserts[0]
    assert "`status`" in insert_sql
    assert "pending_review" in insert_params


def test_events_upload_skeleton_clip_success(monkeypatch, tmp_path: Path):
    fake = _FakeConn(
        responses=[
            {"id": 5, "resident_id": 1, "meta": "{}"},  # existing event
            None,  # UPDATE events SET meta=...
        ]
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_read_clip_privacy_flags", lambda _c, _rid: (True, False))
    monkeypatch.setattr(events_route, "_event_clips_dir", lambda: tmp_path)
    monkeypatch.setattr(events_route, "_cols", lambda _c, _t: {"id", "resident_id", "meta"})

    client = TestClient(app)
    resp = client.post(
        "/api/events/5/skeleton_clip",
        json={
            "resident_id": 1,
            "t_ms": [0, 40],
            "xy": [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.1, 0.1], [1.1, 1.1]],
            ],
            "conf": [[1.0, 1.0], [1.0, 1.0]],
            "mode": "gcn",
            "mc_sigma_tol": 0.03,
            "mc_se_tol": 0.01,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["event_id"] == 5
    files = list(tmp_path.glob("event_5_*.npz"))
    assert files
    with np.load(files[0], allow_pickle=False) as z:
        xy_s = np.asarray(z["xy"], dtype=np.float32)
        conf_s = np.asarray(z["conf"], dtype=np.float32)
        assert bool(np.all((xy_s >= 0.0) & (xy_s <= 1.0)))
        assert bool(np.all((conf_s >= 0.0) & (conf_s <= 1.0)))
    upd = [params for sql, params in fake.executed if "UPDATE events SET `meta`=%s WHERE id=%s" in sql]
    assert upd
    meta = json.loads(upd[0][0])
    assert meta.get("skeleton_clip", {}).get("mc_sigma_tol") == 0.03
    assert meta.get("skeleton_clip", {}).get("mc_se_tol") == 0.01


def test_events_upload_skeleton_clip_v2_payload_json(monkeypatch, tmp_path: Path):
    fake = _FakeConn(
        responses=[
            {"id": 5, "resident_id": 1, "payload_json": "{}"},  # existing event
            None,  # UPDATE events SET payload_json=...
        ]
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_read_clip_privacy_flags", lambda _c, _rid: (True, False))
    monkeypatch.setattr(events_route, "_event_clips_dir", lambda: tmp_path)
    monkeypatch.setattr(events_route, "_cols", lambda _c, _t: {"id", "resident_id", "payload_json"})

    client = TestClient(app)
    resp = client.post(
        "/api/events/5/skeleton_clip",
        json={
            "resident_id": 1,
            "t_ms": [0, 40],
            "xy": [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.1, 0.1], [1.1, 1.1]],
            ],
            "conf": [[1.0, 1.0], [1.0, 1.0]],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert any("UPDATE events SET `payload_json`=%s WHERE id=%s" in sql for sql, _ in fake.executed)


def test_events_upload_skeleton_clip_flat_payload(monkeypatch, tmp_path: Path):
    fake = _FakeConn(
        responses=[
            {"id": 5, "resident_id": 1, "meta": "{}"},
            None,
        ]
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_read_clip_privacy_flags", lambda _c, _rid: (True, False))
    monkeypatch.setattr(events_route, "_event_clips_dir", lambda: tmp_path)
    monkeypatch.setattr(events_route, "_cols", lambda _c, _t: {"id", "resident_id", "meta"})

    client = TestClient(app)
    resp = client.post(
        "/api/events/5/skeleton_clip",
        json={
            "resident_id": 1,
            "t_ms": [0, 40],
            "raw_joints": 2,
            "xy_flat": [0.0, 0.0, 1.0, 1.0, 0.1, 0.1, 1.1, 1.1],
            "conf_flat": [1.0, 1.0, 1.0, 1.0],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["event_id"] == 5
    files = list(tmp_path.glob("event_5_*.npz"))
    assert files
    with np.load(files[0], allow_pickle=False) as z:
        xy_s = np.asarray(z["xy"], dtype=np.float32)
        conf_s = np.asarray(z["conf"], dtype=np.float32)
        assert bool(np.all((xy_s >= 0.0) & (xy_s <= 1.0)))
        assert bool(np.all((conf_s >= 0.0) & (conf_s <= 1.0)))


def test_events_upload_skeleton_clip_rejects_non_monotonic_timestamps(monkeypatch, tmp_path: Path):
    fake = _FakeConn(
        responses=[
            {"id": 5, "resident_id": 1, "meta": "{}"},
        ]
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_read_clip_privacy_flags", lambda _c, _rid: (True, False))
    monkeypatch.setattr(events_route, "_event_clips_dir", lambda: tmp_path)
    monkeypatch.setattr(events_route, "_cols", lambda _c, _t: {"id", "resident_id", "meta"})

    client = TestClient(app)
    resp = client.post(
        "/api/events/5/skeleton_clip",
        json={
            "resident_id": 1,
            "t_ms": [0, 40, 20],
            "xy": [
                [[0.0, 0.0]],
                [[0.1, 0.1]],
                [[0.2, 0.2]],
            ],
            "conf": [[1.0], [1.0], [1.0]],
        },
    )
    assert resp.status_code == 400
    assert "monotonically non-decreasing" in resp.json().get("detail", "")


def test_events_upload_skeleton_clip_rejects_too_many_frames(monkeypatch, tmp_path: Path):
    fake = _FakeConn(
        responses=[
            {"id": 5, "resident_id": 1, "meta": "{}"},
        ]
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_read_clip_privacy_flags", lambda _c, _rid: (True, False))
    monkeypatch.setattr(events_route, "_event_clips_dir", lambda: tmp_path)
    monkeypatch.setattr(events_route, "_cols", lambda _c, _t: {"id", "resident_id", "meta"})

    T = 1300
    t_ms = [float(i * 40) for i in range(T)]
    xy = [[[0.0, 0.0]] for _ in range(T)]
    conf = [[1.0] for _ in range(T)]

    client = TestClient(app)
    resp = client.post(
        "/api/events/5/skeleton_clip",
        json={
            "resident_id": 1,
            "t_ms": t_ms,
            "xy": xy,
            "conf": conf,
        },
    )
    assert resp.status_code == 413
    assert "clip too long" in resp.json().get("detail", "")


def test_events_upload_skeleton_clip_rejects_non_finite_xy(monkeypatch, tmp_path: Path):
    fake = _FakeConn(
        responses=[
            {"id": 5, "resident_id": 1, "meta": "{}"},
        ]
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_read_clip_privacy_flags", lambda _c, _rid: (True, False))
    monkeypatch.setattr(events_route, "_event_clips_dir", lambda: tmp_path)
    monkeypatch.setattr(events_route, "_cols", lambda _c, _t: {"id", "resident_id", "meta"})

    client = TestClient(app)
    resp = client.post(
        "/api/events/5/skeleton_clip",
        json={
            "resident_id": 1,
            "t_ms": [0.0, 40.0],
            "xy_flat": [0.0, 0.0, "nan", 1.0],
            "raw_joints": 1,
            "conf_flat": [1.0, 1.0],
        },
    )
    assert resp.status_code == 400
    assert "xy contains non-finite values" in resp.json().get("detail", "")


def test_events_upload_skeleton_clip_rejects_non_finite_conf(monkeypatch, tmp_path: Path):
    fake = _FakeConn(
        responses=[
            {"id": 5, "resident_id": 1, "meta": "{}"},
        ]
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_read_clip_privacy_flags", lambda _c, _rid: (True, False))
    monkeypatch.setattr(events_route, "_event_clips_dir", lambda: tmp_path)
    monkeypatch.setattr(events_route, "_cols", lambda _c, _t: {"id", "resident_id", "meta"})

    client = TestClient(app)
    resp = client.post(
        "/api/events/5/skeleton_clip",
        json={
            "resident_id": 1,
            "t_ms": [0.0, 40.0],
            "xy_flat": [0.0, 0.0, 1.0, 1.0],
            "raw_joints": 1,
            "conf_flat": [1.0, "nan"],
        },
    )
    assert resp.status_code == 400
    assert "conf contains non-finite values" in resp.json().get("detail", "")


def test_events_upload_skeleton_clip_rejects_t_ms_xy_length_mismatch(monkeypatch, tmp_path: Path):
    fake = _FakeConn(
        responses=[
            {"id": 5, "resident_id": 1, "meta": "{}"},
        ]
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_read_clip_privacy_flags", lambda _c, _rid: (True, False))
    monkeypatch.setattr(events_route, "_event_clips_dir", lambda: tmp_path)
    monkeypatch.setattr(events_route, "_cols", lambda _c, _t: {"id", "resident_id", "meta"})

    client = TestClient(app)
    resp = client.post(
        "/api/events/5/skeleton_clip",
        json={
            "resident_id": 1,
            "t_ms": [0.0, 40.0, 80.0],
            "xy": [
                [[0.0, 0.0]],
                [[0.1, 0.1]],
            ],
            "conf": [[1.0], [1.0]],
        },
    )
    assert resp.status_code == 400
    assert "t_ms and xy time dimensions must match" in resp.json().get("detail", "")


def test_events_upload_skeleton_clip_rejects_conf_xy_length_mismatch(monkeypatch, tmp_path: Path):
    fake = _FakeConn(
        responses=[
            {"id": 5, "resident_id": 1, "meta": "{}"},
        ]
    )
    monkeypatch.setattr(events_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(events_route, "_read_clip_privacy_flags", lambda _c, _rid: (True, False))
    monkeypatch.setattr(events_route, "_event_clips_dir", lambda: tmp_path)
    monkeypatch.setattr(events_route, "_cols", lambda _c, _t: {"id", "resident_id", "meta"})

    client = TestClient(app)
    resp = client.post(
        "/api/events/5/skeleton_clip",
        json={
            "resident_id": 1,
            "t_ms": [0.0, 40.0, 80.0],
            "xy": [
                [[0.0, 0.0]],
                [[0.1, 0.1]],
                [[0.2, 0.2]],
            ],
            "conf": [[1.0], [1.0]],
        },
    )
    assert resp.status_code == 400
    assert "conf and xy time dimensions must match" in resp.json().get("detail", "")


def test_settings_get_system_settings_branch(monkeypatch):
    fake = _FakeConn(
        responses=[
            {
                "monitoring_enabled": 1,
                "camera_source": "webcam",
                "active_model_code": "TCN",
                "fall_threshold": 0.81,
                "alert_cooldown_sec": 4,
                "mc_enabled": 1,
                "active_dataset_code": "le2i",
                "active_op_code": "OP-1",
                "active_operating_point_id": 4,
            }
        ]
    )
    monkeypatch.setattr(settings_route, "get_conn", lambda: _cm_conn(fake))
    monkeypatch.setattr(settings_route, "_ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(settings_route, "_table_exists", lambda _c, t: t == "system_settings")
    monkeypatch.setattr(settings_route, "_derive_ops_params_from_yaml", lambda **_k: {"ui": {"op_code": "OP-1", "tau_low": 0.1, "tau_high": 0.81, "cooldown_s": 4}})

    client = TestClient(app)
    resp = client.get("/api/settings?resident_id=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["system"]["active_model_code"] == "TCN"
    assert body["system"]["active_dataset_code"] == "le2i"


def test_dashboard_summary_model_lookup_and_fall_events(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"monitoring_enabled": 1, "active_model_code": "HYBRID", "active_model_id": 2},
            {"name": "Graph Model"},
            {"falls_detected": 1, "false_alarms": 2},
        ]
    )
    monkeypatch.setattr(dashboard_route, "get_conn", lambda: _cm_conn(fake))
    monkeypatch.setattr(dashboard_route, "_resident_exists", lambda _c, _rid: True)
    monkeypatch.setattr(dashboard_route, "_one_resident_id", lambda _c: 1)
    monkeypatch.setattr(dashboard_route, "_table_exists", lambda _c, t: t in {"system_settings", "models", "fall_events"})
    monkeypatch.setattr(dashboard_route, "_col_exists", lambda _c, _t, _col: False)

    client = TestClient(app)
    resp = client.get("/api/dashboard/summary?resident_id=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["system"]["model_name"] == "Graph Model"
    assert body["today"]["falls_detected"] == 1
    assert body["today"]["false_alarms"] == 2


def test_specs_models_summary_bad_fps_default(monkeypatch):
    spec = SimpleNamespace(
        dataset="le2i",
        arch="tcn",
        ckpt="/tmp/x.pt",
        data_cfg={"fps_default": "bad-number"},
        ops={},
        alert_cfg={},
    )
    monkeypatch.setattr(specs_route, "_get_deploy_specs", lambda: {"le2i_tcn": spec})
    monkeypatch.setattr(specs_route, "get_conn", lambda: _cm_conn(_FakeConn(responses=[[]])))

    client = TestClient(app)
    resp = client.get("/api/models/summary")
    assert resp.status_code == 200
    assert resp.json()["models"][0]["fps_default"] is None


def test_caregivers_update_existing_branch(monkeypatch):
    fake = _FakeConn(
        responses=[
            {"id": 2},  # existing caregiver
            None,       # update
            {"id": 2, "resident_id": 1, "name": "Care D", "email": "d@x", "phone": "4"},
        ]
    )
    monkeypatch.setattr(caregivers_route, "get_conn_optional", lambda: _cm_conn(fake))
    monkeypatch.setattr(caregivers_route, "_ensure_caregivers_table", lambda _c: None)
    monkeypatch.setattr(caregivers_route, "_table_exists", lambda _c, t: t == "caregivers")

    client = TestClient(app)
    resp = client.put("/api/caregivers", json={"resident_id": 1, "name": "Care D"})
    assert resp.status_code == 200
    assert resp.json()["caregiver"]["name"] == "Care D"


def test_operating_points_v1_db_path(monkeypatch):
    fake = _FakeConn(
        responses=[
            [
                {
                    "id": 4,
                    "model_code": "GCN",
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
    monkeypatch.setattr(ops_route, "_ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(ops_route, "_detect_variants", lambda _c: {"settings": "v1", "events": "v1", "ops": "v1"})
    monkeypatch.setattr(ops_route, "_table_exists", lambda _c, _t: False)

    client = TestClient(app)
    resp = client.get("/api/operating_points?model_code=GCN&dataset_code=muvim")
    assert resp.status_code == 200
    body = resp.json()
    assert body["db_available"] is True
    assert body["operating_points"][0]["threshold_high"] == 0.8


def test_operating_points_v2_unknown_model_404(monkeypatch):
    fake = _FakeConn(responses=[{}])  # model lookup returns empty
    monkeypatch.setattr(ops_route, "get_conn", lambda: _cm_conn(fake))
    monkeypatch.setattr(ops_route, "_ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(ops_route, "_detect_variants", lambda _c: {"settings": "v2", "events": "v2", "ops": "v2"})
    monkeypatch.setattr(ops_route, "_table_exists", lambda _c, t: t == "models")

    client = TestClient(app)
    resp = client.get("/api/operating_points?model_code=GCN")
    assert resp.status_code == 404
