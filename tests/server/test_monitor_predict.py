from contextlib import contextmanager
from types import SimpleNamespace
import json
import math

import numpy as np
from fastapi.testclient import TestClient

from server.main import app
from server.routes import monitor as monitor_route


class _DummyTracker:
    def __init__(self, _cfg):
        self._calls = 0

    def step(self, p, t_s):
        self._calls += 1
        state = "fall" if float(p) >= 0.8 else "not_fall"
        return SimpleNamespace(
            triage_state=state,
            ps=float(p),
            p_in=float(p),
            cooldown_remaining_s=0.0,
            started_event=(self._calls == 1 and state == "fall"),
        )


class _DummyUncertainTracker:
    def __init__(self, _cfg):
        self._calls = 0

    def step(self, p, t_s):
        self._calls += 1
        return SimpleNamespace(
            triage_state="uncertain",
            ps=float(p),
            p_in=float(p),
            cooldown_remaining_s=0.0,
            started_event=False,
        )


def _mock_predict_spec(*, spec_key, **_kwargs):
    mu = 0.91 if spec_key.endswith("_tcn") else 0.88
    return {
        "mu": mu,
        "p_det": mu,
        "sigma": 0.05,
        "mc_n_used": 8,
        "tau_low": 0.2,
        "tau_high": 0.8,
        "alert_cfg": {"tau_low": 0.2, "tau_high": 0.8, "k": 2, "n": 3, "cooldown_s": 0.0},
    }


def _specs():
    return {"caucafall_tcn": object(), "caucafall_gcn": object()}


def test_predict_window_dual_happy_path(monkeypatch):
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", _specs)
    monkeypatch.setattr(monitor_route, "_predict_spec", _mock_predict_spec)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-dual",
            "mode": "dual",
            "dataset_code": "caucafall",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
            "persist": False,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["triage_state"] in {"fall", "uncertain"}
    assert body["dataset_code"] == "caucafall"
    assert "tcn" in body["models"]
    assert "gcn" in body["models"]


def test_predict_window_falls_back_to_defaults_when_db_unavailable(monkeypatch):
    @contextmanager
    def _broken_conn():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr(monitor_route, "get_conn", _broken_conn)
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", _specs)
    monkeypatch.setattr(monitor_route, "_predict_spec", _mock_predict_spec)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-fallback",
            "mode": "tcn",
            "dataset_code": "caucafall",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_code"] == "caucafall"
    assert body["event_id"] is None


def test_predict_window_accepts_raw_payload(monkeypatch):
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", _specs)
    monkeypatch.setattr(monitor_route, "_get_pose_preprocess_cfg", lambda _spec_key: {"smooth_k": 5})
    monkeypatch.setattr(monitor_route, "_predict_spec", _mock_predict_spec)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-raw",
            "mode": "tcn",
            "dataset_code": "caucafall",
            "raw_t_ms": [0.0, 40.0],
            "raw_xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "raw_conf": [[1.0], [1.0]],
            "target_T": 2,
            "window_end_t_ms": 40.0,
        },
    )
    assert resp.status_code == 200


def test_predict_window_raw_path_works_without_window_end_timestamp(monkeypatch):
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", _specs)
    monkeypatch.setattr(monitor_route, "_get_pose_preprocess_cfg", lambda _spec_key: {"smooth_k": 5})
    monkeypatch.setattr(monitor_route, "_predict_spec", _mock_predict_spec)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-raw-no-window-end",
            "mode": "tcn",
            "dataset_code": "caucafall",
            "raw_t_ms": [0.0, 40.0],
            "raw_xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "raw_conf": [[1.0], [1.0]],
            "target_T": 2,
        },
    )

    assert resp.status_code == 200


def test_predict_window_raw_path_uses_spec_pose_preprocess_cfg(monkeypatch):
    seen = {"cfg": None}

    def _capture_preprocess(xy, conf, *, cfg=None):
        seen["cfg"] = dict(cfg or {})
        return xy, conf

    monkeypatch.setattr(monitor_route, "_get_deploy_specs", _specs)
    monkeypatch.setattr(
        monitor_route,
        "_get_pose_preprocess_cfg",
        lambda _spec_key: {"smooth_k": 7, "normalize": "shoulder", "pelvis_fill": "forward", "rotate": "shoulders"},
    )
    monkeypatch.setattr(monitor_route, "_preprocess_online_raw_window", _capture_preprocess)
    monkeypatch.setattr(monitor_route, "_predict_spec", _mock_predict_spec)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-raw-preproc-cfg",
            "mode": "tcn",
            "dataset_code": "caucafall",
            "raw_t_ms": [0.0, 40.0],
            "raw_xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "raw_conf": [[1.0], [1.0]],
            "target_T": 2,
            "window_end_t_ms": 40.0,
        },
    )

    assert resp.status_code == 200
    assert seen["cfg"]["smooth_k"] == 7
    assert seen["cfg"]["normalize"] == "shoulder"
    assert seen["cfg"]["pelvis_fill"] == "forward"
    assert seen["cfg"]["rotate"] == "shoulders"


def test_predict_window_replay_raw_path_uses_measured_fps(monkeypatch):
    seen = {"fps": None}

    def _capture_predict(*, fps=None, spec_key=None, **_kwargs):
        seen["fps"] = float(fps)
        return _mock_predict_spec(spec_key=spec_key or "caucafall_tcn")

    monkeypatch.setattr(monitor_route, "_get_deploy_specs", lambda: {"le2i_tcn": object()})
    monkeypatch.setattr(monitor_route, "_get_pose_preprocess_cfg", lambda _spec_key: {"smooth_k": 5})
    monkeypatch.setattr(monitor_route, "_predict_spec", _capture_predict)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-replay-fps",
            "mode": "tcn",
            "dataset_code": "le2i",
            "target_T": 2,
            "input_source": "replay",
            "window_end_t_ms": 66.6666667,
            "raw_t_ms": [0.0, 66.6666667],
            "raw_xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "raw_conf": [[1.0], [1.0]],
        },
    )

    assert resp.status_code == 200
    assert math.isclose(float(seen["fps"]), 15.0, rel_tol=1e-6, abs_tol=1e-6)


def test_predict_window_forwards_mc_tolerances(monkeypatch):
    seen = {"vals": []}

    def _capture_predict(*, spec_key=None, **_kwargs):
        seen["vals"].append(spec_key)
        return _mock_predict_spec(spec_key=spec_key or "caucafall_tcn")

    monkeypatch.setattr(monitor_route, "_get_deploy_specs", _specs)
    monkeypatch.setattr(monitor_route, "_predict_spec", _capture_predict)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-mc-tol",
            "mode": "dual",
            "dataset_code": "caucafall",
            "mc_sigma_tol": 0.03,
            "mc_se_tol": 0.01,
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
        },
    )
    assert resp.status_code == 200
    assert len(seen["vals"]) == 2


def test_predict_window_sanitizes_direct_xy_conf_before_predict(monkeypatch):
    seen = {"xy": None, "conf": None}

    def _capture_predict(*, joints_xy=None, conf=None, spec_key=None, **_kwargs):
        seen["xy"] = joints_xy
        seen["conf"] = conf
        return _mock_predict_spec(spec_key=spec_key or "caucafall_tcn")

    monkeypatch.setattr(monitor_route, "_get_deploy_specs", _specs)
    monkeypatch.setattr(monitor_route, "_predict_spec", _capture_predict)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-sanitize-direct",
            "mode": "tcn",
            "dataset_code": "caucafall",
            "xy": [[[-0.5, 1.5]], [[2.0, -1.0]]],
            "conf": [[1.2], ["nan"]],
            "target_T": 2,
        },
    )
    assert resp.status_code == 200
    assert isinstance(seen["xy"], list)
    assert isinstance(seen["conf"], list)


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

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        return None


def _conn_cm(conn):
    @contextmanager
    def _cm():
        yield conn

    return _cm()


def test_predict_window_uses_db_settings_and_persists_event(monkeypatch):
    cfg_conn = _FakeConn(
        responses=[
            {
                "active_dataset_code": "caucafall",
                "mc_enabled": 0,
                "mc_M": 7,
                "active_model_code": "TCN",
                "active_op_code": None,
                "active_operating_point_id": 3,
            },
            {"code": "OP-3"},
        ]
    )
    ins_conn = _FakeConn(lastrowid=42)
    conns = [cfg_conn, ins_conn]

    monkeypatch.setattr(monitor_route, "get_conn", lambda: _conn_cm(conns.pop(0)))
    monkeypatch.setattr(monitor_route, "_detect_variants", lambda _c: {"settings": "v2", "events": "v1", "ops": "v1"})
    monkeypatch.setattr(monitor_route, "_ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(monitor_route, "_table_exists", lambda _c, t: t in {"system_settings", "operating_points", "events"})
    monkeypatch.setattr(
        monitor_route.core,
        "_cols",
        lambda _c, _t: {"resident_id", "type", "severity", "model_code", "operating_point_id", "score", "meta", "ts"},
    )
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", _specs)
    monkeypatch.setattr(monitor_route, "_predict_spec", _mock_predict_spec)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-db",
            "mode": "tcn",
            "input_source": "video",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
            "persist": True,
            "resident_id": 1,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["op_code"] == "OP-3"
    assert body["use_mc"] is False
    assert body["event_id"] == 42


def test_predict_window_persists_replay_uncertain_event(monkeypatch):
    cfg_conn = _FakeConn(
        responses=[
            {
                "active_dataset_code": "caucafall",
                "mc_enabled": 0,
                "mc_M": 7,
                "active_model_code": "TCN",
                "active_op_code": None,
                "active_operating_point_id": 2,
            },
            {"code": "OP-2"},
        ]
    )
    ins_conn = _FakeConn(lastrowid=77)
    conns = [cfg_conn, ins_conn]

    monkeypatch.setattr(monitor_route, "get_conn", lambda: _conn_cm(conns.pop(0)))
    monkeypatch.setattr(monitor_route, "_detect_variants", lambda _c: {"settings": "v2", "events": "v1", "ops": "v1"})
    monkeypatch.setattr(monitor_route, "_ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(monitor_route, "_table_exists", lambda _c, t: t in {"system_settings", "operating_points", "events"})
    monkeypatch.setattr(
        monitor_route.core,
        "_cols",
        lambda _c, _t: {"resident_id", "type", "severity", "model_code", "operating_point_id", "score", "meta", "ts"},
    )
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", _specs)
    monkeypatch.setattr(monitor_route, "_predict_spec", _mock_predict_spec)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyUncertainTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-replay-uncertain",
            "mode": "tcn",
            "input_source": "video",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
            "persist": True,
            "resident_id": 1,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["triage_state"] == "uncertain"
    assert body["event_id"] == 77


def test_resample_pose_window_basic_path():
    xy_out, conf_out, start_ms, end_ms, cap_fps = monitor_route._resample_pose_window(
        raw_t_ms=[0, 40, 80],
        raw_xy=[
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.5, 0.5], [1.5, 1.5]],
            [[1.0, 1.0], [2.0, 2.0]],
        ],
        raw_conf=[[1.0, 1.0], [0.9, 0.9], [0.8, 0.8]],
        target_fps=25.0,
        target_T=3,
    )
    assert len(xy_out) == 3
    assert len(conf_out) == 3
    assert end_ms >= start_ms
    assert cap_fps is not None


def test_resample_pose_window_rejects_bad_input():
    xy_out, conf_out, *_ = monitor_route._resample_pose_window(raw_t_ms="bad", raw_xy=[], raw_conf=None)
    assert xy_out == []
    assert conf_out == []


def test_resample_pose_window_rejects_numpy_timestamps():
    raw_t_ms = np.array([0.0, 40.0, 80.0], dtype=np.float32)
    xy_out, conf_out, *_ = monitor_route._resample_pose_window(
        raw_t_ms=raw_t_ms,
        raw_xy=[[[0.0, 0.0]], [[0.5, 0.5]], [[1.0, 1.0]]],
        raw_conf=[[1.0], [0.9], [0.8]],
        target_fps=25.0,
        target_T=3,
    )
    assert xy_out == []
    assert conf_out == []


def test_resample_pose_window_filters_non_monotonic_timestamps():
    xy_out, conf_out, *_ = monitor_route._resample_pose_window(
        raw_t_ms=[0.0, 40.0, 40.0, 80.0],
        raw_xy=[[[0.0, 0.0]], [[0.5, 0.5]], [[0.7, 0.7]], [[1.0, 1.0]]],
        raw_conf=[[1.0], [0.9], [0.8], [0.7]],
        target_fps=25.0,
        target_T=3,
        window_end_t_ms=80.0,
    )
    assert len(xy_out) == 3
    assert math.isclose(float(xy_out[0][0][0]), 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(xy_out[1][0][0]), 0.5, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(xy_out[2][0][0]), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert len(conf_out) == 3


def test_resample_pose_window_aligned_values():
    xy_out, conf_out, *_ = monitor_route._resample_pose_window(
        raw_t_ms=[0.0, 40.0],
        raw_xy=[[[0.0, 0.0], [1.0, 1.0]], [[0.1, 0.2], [1.1, 1.2]]],
        raw_conf=[[0.9, 0.8], [0.7, 0.6]],
        target_fps=25.0,
        target_T=2,
        window_end_t_ms=40.0,
    )
    assert len(xy_out) == 2
    assert len(conf_out) == 2
    assert math.isclose(float(xy_out[1][1][1]), 1.2, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(conf_out[1][0]), 0.7, rel_tol=1e-6, abs_tol=1e-6)


def test_resample_pose_window_fallback_preserves_original_indices():
    xy_out, conf_out, *_ = monitor_route._resample_pose_window(
        raw_t_ms=[0.0, 40.0, 80.0],
        raw_xy=[
            [[0.0, 0.0], [0.0, 0.0]],
            "bad_frame",
            [[1.0, 1.0, 9.0], [1.0, 1.0]],
        ],
        raw_conf=[[0.2, 0.2], [0.3], [0.8, 0.8]],
        target_fps=25.0,
        target_T=3,
        window_end_t_ms=40.0,
    )
    assert len(xy_out) == 3
    assert len(conf_out) == 3
    assert math.isclose(float(xy_out[2][0][0]), 0.5, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(conf_out[2][0]), 0.5, rel_tol=1e-6, abs_tol=1e-6)
