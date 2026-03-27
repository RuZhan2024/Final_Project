from contextlib import contextmanager
from types import SimpleNamespace
import math
import json
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
    if spec_key.endswith("_tcn"):
        mu = 0.91
    else:
        mu = 0.88
    return {
        "mu": mu,
        "p_det": mu,
        "sigma": 0.05,
        "mc_n_used": 8,
        "tau_low": 0.2,
        "tau_high": 0.8,
        "alert_cfg": {"tau_low": 0.2, "tau_high": 0.8, "k": 2, "n": 3, "cooldown_s": 0.0},
    }


def test_predict_window_dual_happy_path(monkeypatch):
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", lambda: {"muvim_tcn": object(), "muvim_gcn": object()})
    monkeypatch.setattr(monitor_route, "_predict_spec", _mock_predict_spec)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-dual",
            "mode": "dual",
            "dataset_code": "muvim",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
            "persist": False,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["triage_state"] == "fall"
    assert body["dataset_code"] == "muvim"
    assert body["op_code"] == "OP-2"
    assert body["use_mc"] is True
    assert "tcn" in body["models"]
    assert "gcn" in body["models"]
    assert "hybrid" in body["models"]
    assert body["mc_n_used"]["tcn"] == 8
    assert body["mc_n_used"]["gcn"] == 8


def test_predict_window_falls_back_to_defaults_when_db_unavailable(monkeypatch):
    @contextmanager
    def _broken_conn():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr(monitor_route, "get_conn", _broken_conn)
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", lambda: {"muvim_tcn": object(), "muvim_gcn": object()})
    monkeypatch.setattr(monitor_route, "_predict_spec", _mock_predict_spec)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-fallback",
            "mode": "tcn",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_code"] == "muvim"
    assert body["op_code"] == "OP-2"
    assert body["use_mc"] is True
    assert body["event_id"] is None


def test_predict_window_accepts_flat_raw_payload(monkeypatch):
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", lambda: {"muvim_tcn": object(), "muvim_gcn": object()})
    monkeypatch.setattr(monitor_route, "_predict_spec", _mock_predict_spec)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-flat",
            "mode": "tcn",
            "dataset_code": "muvim",
            "raw_t_ms": [0.0, 40.0],
            "raw_joints": 1,
            "raw_xy_flat": [0.0, 0.0, 0.1, 0.1],
            "raw_conf_flat": [1.0, 1.0],
            "target_T": 2,
            "window_end_t_ms": 40.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["triage_state"] in {"fall", "not_fall", "uncertain"}


def test_predict_window_forwards_mc_tolerances(monkeypatch):
    seen = {"vals": []}

    def _capture_predict(*, mc_sigma_tol=None, mc_se_tol=None, spec_key=None, **_kwargs):
        seen["vals"].append((spec_key, mc_sigma_tol, mc_se_tol))
        return _mock_predict_spec(spec_key=spec_key or "muvim_tcn")

    monkeypatch.setattr(monitor_route, "_get_deploy_specs", lambda: {"muvim_tcn": object(), "muvim_gcn": object()})
    monkeypatch.setattr(monitor_route, "_predict_spec", _capture_predict)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-mc-tol",
            "mode": "dual",
            "dataset_code": "muvim",
            "mc_sigma_tol": 0.03,
            "mc_se_tol": 0.01,
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mc_sigma_tol"] == 0.03
    assert body["mc_se_tol"] == 0.01
    assert len(seen["vals"]) == 2
    assert all(v[1] == 0.03 and v[2] == 0.01 for v in seen["vals"])


def test_predict_window_sanitizes_direct_xy_conf_before_predict(monkeypatch):
    seen = {"xy": None, "conf": None}

    def _capture_predict(*, joints_xy=None, conf=None, spec_key=None, **_kwargs):
        seen["xy"] = joints_xy
        seen["conf"] = conf
        return _mock_predict_spec(spec_key=spec_key or "muvim_tcn")

    monkeypatch.setattr(monitor_route, "_get_deploy_specs", lambda: {"muvim_tcn": object(), "muvim_gcn": object()})
    monkeypatch.setattr(monitor_route, "_predict_spec", _capture_predict)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-sanitize-direct",
            "mode": "tcn",
            "dataset_code": "muvim",
            "xy": [[[-0.5, 1.5]], [[2.0, -1.0]]],
            "conf": [[1.2], ["nan"]],
            "target_T": 2,
        },
    )
    assert resp.status_code == 200
    assert isinstance(seen["xy"], np.ndarray)
    assert isinstance(seen["conf"], np.ndarray)
    assert bool(np.isfinite(seen["xy"]).all())
    assert bool(np.isfinite(seen["conf"]).all())
    assert bool(np.all((seen["xy"] >= 0.0) & (seen["xy"] <= 1.0)))
    assert bool(np.all((seen["conf"] >= 0.0) & (seen["conf"] <= 1.0)))


def test_predict_window_skips_db_when_runtime_fields_present(monkeypatch):
    def _boom_conn():
        raise AssertionError("DB should not be touched for fully-specified runtime payload")

    monkeypatch.setattr(monitor_route, "get_conn", _boom_conn)
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", lambda: {"muvim_tcn": object(), "muvim_gcn": object()})
    monkeypatch.setattr(monitor_route, "_predict_spec", _mock_predict_spec)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-nodb",
            "mode": "tcn",
            "dataset_code": "muvim",
            "op_code": "OP-2",
            "use_mc": True,
            "mc_M": 8,
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
        },
    )
    assert resp.status_code == 200


def test_predict_window_maps_predict_value_error_to_400(monkeypatch):
    def _raise_bad_input(**_kwargs):
        raise ValueError("bad shape in runtime")

    monkeypatch.setattr(monitor_route, "_get_deploy_specs", lambda: {"muvim_tcn": object(), "muvim_gcn": object()})
    monkeypatch.setattr(monitor_route, "_predict_spec", _raise_bad_input)
    monkeypatch.setattr(monitor_route, "OnlineAlertTracker", _DummyTracker)

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "session_id": "s-bad-runtime",
            "mode": "tcn",
            "dataset_code": "muvim",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
        },
    )
    assert resp.status_code == 400
    assert "invalid inference input" in resp.json().get("detail", "")


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
    xy_out, conf_out, *_ = monitor_route._resample_pose_window(
        raw_t_ms="bad",
        raw_xy=[],
        raw_conf=None,
    )
    assert xy_out == []
    assert conf_out == []


def test_resample_pose_window_accepts_numpy_timestamps():
    raw_t_ms = np.array([0.0, 40.0, 80.0], dtype=np.float32)
    xy_out, conf_out, start_ms, end_ms, cap_fps = monitor_route._resample_pose_window(
        raw_t_ms=raw_t_ms,
        raw_xy=[
            [[0.0, 0.0]],
            [[0.5, 0.5]],
            [[1.0, 1.0]],
        ],
        raw_conf=[[1.0], [0.9], [0.8]],
        target_fps=25.0,
        target_T=3,
    )
    assert hasattr(xy_out, "shape")
    assert xy_out.shape == (3, 1, 2)
    assert conf_out.shape == (3, 1)
    assert end_ms >= start_ms
    assert cap_fps is not None


def test_resample_pose_window_flat_payload_path():
    raw_t_ms = [0.0, 40.0, 80.0]
    raw_joints = 2
    # frame0: (0,0),(1,1) ; frame1: (0.5,0.5),(1.5,1.5) ; frame2: (1,1),(2,2)
    raw_xy_flat = [
        0.0, 0.0, 1.0, 1.0,
        0.5, 0.5, 1.5, 1.5,
        1.0, 1.0, 2.0, 2.0,
    ]
    raw_conf_flat = [
        1.0, 1.0,
        0.9, 0.9,
        0.8, 0.8,
    ]
    xy_out, conf_out, start_ms, end_ms, cap_fps = monitor_route._resample_pose_window(
        raw_t_ms=raw_t_ms,
        raw_xy=[],
        raw_conf=None,
        raw_xy_flat=raw_xy_flat,
        raw_conf_flat=raw_conf_flat,
        raw_joints=raw_joints,
        target_fps=25.0,
        target_T=3,
    )
    assert hasattr(xy_out, "shape")
    assert xy_out.shape == (3, 2, 2)
    assert conf_out.shape == (3, 2)
    assert end_ms >= start_ms
    assert cap_fps is not None


def test_resample_pose_window_prevalidated_matches_default_path():
    raw_t_ms = np.array([0.0, 40.0, 80.0], dtype=np.float32)
    raw_joints = 2
    raw_xy_flat = np.array(
        [
            0.0, 0.0, 1.0, 1.0,
            0.5, 0.5, 1.5, 1.5,
            1.0, 1.0, 2.0, 2.0,
        ],
        dtype=np.float32,
    )
    raw_conf_flat = np.array(
        [
            1.0, 1.0,
            0.9, 0.9,
            0.8, 0.8,
        ],
        dtype=np.float32,
    )
    base = monitor_route._resample_pose_window(
        raw_t_ms=raw_t_ms,
        raw_xy=[],
        raw_conf=None,
        raw_xy_flat=raw_xy_flat,
        raw_conf_flat=raw_conf_flat,
        raw_joints=raw_joints,
        target_fps=25.0,
        target_T=3,
        window_end_t_ms=80.0,
        prevalidated_raw=False,
    )
    fast = monitor_route._resample_pose_window(
        raw_t_ms=raw_t_ms,
        raw_xy=[],
        raw_conf=None,
        raw_xy_flat=raw_xy_flat,
        raw_conf_flat=raw_conf_flat,
        raw_joints=raw_joints,
        target_fps=25.0,
        target_T=3,
        window_end_t_ms=80.0,
        prevalidated_raw=True,
    )
    assert np.array_equal(base[0], fast[0])
    assert np.array_equal(base[1], fast[1])
    assert math.isclose(float(base[2]), float(fast[2]), rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(float(base[3]), float(fast[3]), rel_tol=1e-9, abs_tol=1e-9)
    if base[4] is None or fast[4] is None:
        assert base[4] is fast[4]
    else:
        assert math.isclose(float(base[4]), float(fast[4]), rel_tol=1e-9, abs_tol=1e-9)


def test_resample_pose_window_flat_payload_filters_non_monotonic_and_nan():
    raw_t_ms = [0.0, float("nan"), 40.0, 40.0, 80.0]
    raw_joints = 1
    # one joint per frame: (x,y)
    raw_xy_flat = [
        0.0, 0.0,
        9.0, 9.0,   # dropped (nan t)
        0.5, 0.5,
        0.7, 0.7,   # dropped (duplicate t)
        1.0, 1.0,
    ]
    raw_conf_flat = [1.0, 0.1, 0.9, 0.8, 0.7]

    xy_out, conf_out, *_ = monitor_route._resample_pose_window(
        raw_t_ms=raw_t_ms,
        raw_xy=[],
        raw_conf=None,
        raw_xy_flat=raw_xy_flat,
        raw_conf_flat=raw_conf_flat,
        raw_joints=raw_joints,
        target_fps=25.0,
        target_T=3,
        window_end_t_ms=80.0,
    )

    assert xy_out.shape == (3, 1, 2)
    assert conf_out.shape == (3, 1)
    # Should interpolate from kept frames [0ms, 40ms, 80ms], ignoring dropped entries.
    assert math.isclose(float(xy_out[0, 0, 0]), 0.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(xy_out[1, 0, 0]), 0.5, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(xy_out[2, 0, 0]), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_resample_pose_window_aligned_fast_path_values():
    raw_t_ms = [0.0, 40.0]
    raw_xy = [
        [[0.0, 0.0], [1.0, 1.0]],
        [[0.1, 0.2], [1.1, 1.2]],
    ]
    raw_conf = [
        [0.9, 0.8],
        [0.7, 0.6],
    ]
    xy_out, conf_out, *_ = monitor_route._resample_pose_window(
        raw_t_ms=raw_t_ms,
        raw_xy=raw_xy,
        raw_conf=raw_conf,
        target_fps=25.0,  # dt=40ms -> aligned with raw_t_ms
        target_T=2,
        window_end_t_ms=40.0,
    )
    assert hasattr(xy_out, "shape")
    assert xy_out.shape == (2, 2, 2)
    assert conf_out.shape == (2, 2)
    assert math.isclose(float(xy_out[1, 1, 1]), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(conf_out[1, 0]), 0.7, rel_tol=1e-6, abs_tol=1e-6)


def test_resample_pose_window_clamps_xy_conf_to_unit_interval():
    xy_out, conf_out, *_ = monitor_route._resample_pose_window(
        raw_t_ms=[0.0, 40.0],
        raw_xy=[
            [[-0.2, 1.3]],
            [[1.5, -0.4]],
        ],
        raw_conf=[[1.3], [-0.5]],
        target_fps=25.0,
        target_T=2,
        window_end_t_ms=40.0,
    )
    assert bool(np.all((xy_out >= 0.0) & (xy_out <= 1.0)))
    assert bool(np.all((conf_out >= 0.0) & (conf_out <= 1.0)))


def test_resample_pose_window_clamps_interpolated_xy_conf():
    xy_out, conf_out, *_ = monitor_route._resample_pose_window(
        raw_t_ms=[0.0, 80.0],
        raw_xy=[
            [[-1.0, 2.0]],
            [[2.0, -1.0]],
        ],
        raw_conf=[[2.0], [-1.0]],
        target_fps=25.0,
        target_T=3,
        window_end_t_ms=80.0,
    )
    assert bool(np.all((xy_out >= 0.0) & (xy_out <= 1.0)))
    assert bool(np.all((conf_out >= 0.0) & (conf_out <= 1.0)))


def test_resample_pose_window_sanitizes_non_finite_xy_conf():
    raw_t_ms = [0.0, 40.0, 80.0]
    raw_joints = 1
    raw_xy_flat = [
        float("nan"), 0.0,
        0.5, float("inf"),
        float("-inf"), 1.0,
    ]
    raw_conf_flat = [
        1.0,
        float("nan"),
        float("inf"),
    ]
    xy_out, conf_out, *_ = monitor_route._resample_pose_window(
        raw_t_ms=raw_t_ms,
        raw_xy=[],
        raw_conf=None,
        raw_xy_flat=raw_xy_flat,
        raw_conf_flat=raw_conf_flat,
        raw_joints=raw_joints,
        target_fps=25.0,
        target_T=3,
        window_end_t_ms=80.0,
    )
    assert xy_out.shape == (3, 1, 2)
    assert conf_out.shape == (3, 1)
    assert bool(np.isfinite(xy_out).all())
    assert bool(np.isfinite(conf_out).all())


def test_resample_pose_window_fallback_preserves_original_indices():
    # Middle frame is malformed and should be filtered out; fallback must still
    # map to the original kept source indices (0 and 2), not compacted [0,1].
    xy_out, conf_out, *_ = monitor_route._resample_pose_window(
        raw_t_ms=[0.0, 40.0, 80.0],
        raw_xy=[
            [[0.0, 0.0], [0.0, 0.0]],
            "bad_frame",
            [[1.0, 1.0, 9.0], [1.0, 1.0]],  # ragged point dims force fallback path
        ],
        raw_conf=[
            [0.2, 0.2],
            [0.3],
            [0.8, 0.8],
        ],
        target_fps=25.0,
        target_T=3,
        window_end_t_ms=40.0,
    )
    assert hasattr(xy_out, "shape")
    assert xy_out.shape == (3, 2, 2)
    assert conf_out.shape == (3, 2)
    # Last frame should interpolate between original source indices 0 and 2.
    assert math.isclose(float(xy_out[2, 0, 0]), 0.5, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(conf_out[2, 0]), 0.5, rel_tol=1e-6, abs_tol=1e-6)


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
                "active_dataset_code": "muvim",
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

    def _get_conn():
        return _conn_cm(conns.pop(0))

    monkeypatch.setattr(monitor_route, "get_conn", _get_conn)
    monkeypatch.setattr(monitor_route, "_detect_variants", lambda _c: {"settings": "v2", "events": "v1", "ops": "v1"})
    monkeypatch.setattr(monitor_route, "_ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(monitor_route, "_table_exists", lambda _c, t: t in {"system_settings", "operating_points", "events"})
    monkeypatch.setattr(
        monitor_route.core,
        "_cols",
        lambda _c, _t: {"resident_id", "type", "severity", "model_code", "operating_point_id", "score", "meta", "ts"},
    )
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", lambda: {"muvim_tcn": object(), "muvim_gcn": object()})
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
            "mc_sigma_tol": 0.025,
            "mc_se_tol": 0.012,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["op_code"] == "OP-3"
    assert body["use_mc"] is False
    assert body["event_id"] == 42

    insert_params = None
    for sql, params in ins_conn.executed:
        if "INSERT INTO events" in sql:
            insert_params = params
            break
    assert insert_params is not None
    assert insert_params[1] == "fall"
    assert insert_params[2] == "high"
    meta_json = None
    for v in insert_params:
        if isinstance(v, str) and "\"models\"" in v and "\"dataset\"" in v:
            meta_json = v
            break
    assert meta_json is not None
    meta = json.loads(meta_json)
    assert meta.get("mc_sigma_tol") == 0.025
    assert meta.get("mc_se_tol") == 0.012
    assert meta.get("input_source") == "video"
    assert meta.get("persist_event_type") == "fall"


def test_predict_window_persists_replay_uncertain_event(monkeypatch):
    cfg_conn = _FakeConn(
        responses=[
            {
                "active_dataset_code": "muvim",
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

    def _get_conn():
        return _conn_cm(conns.pop(0))

    monkeypatch.setattr(monitor_route, "get_conn", _get_conn)
    monkeypatch.setattr(monitor_route, "_detect_variants", lambda _c: {"settings": "v2", "events": "v1", "ops": "v1"})
    monkeypatch.setattr(monitor_route, "_ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(monitor_route, "_table_exists", lambda _c, t: t in {"system_settings", "operating_points", "events"})
    monkeypatch.setattr(
        monitor_route.core,
        "_cols",
        lambda _c, _t: {"resident_id", "type", "severity", "model_code", "operating_point_id", "score", "meta", "ts"},
    )
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", lambda: {"muvim_tcn": object(), "muvim_gcn": object()})
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
    assert body["notification_dispatch"] is None

    insert_params = None
    for sql, params in ins_conn.executed:
        if "INSERT INTO events" in sql:
            insert_params = params
            break
    assert insert_params is not None
    assert insert_params[1] == "uncertain"
    assert insert_params[2] == "medium"
    meta = json.loads(insert_params[-1])
    assert meta.get("input_source") == "video"
    assert meta.get("persist_event_type") == "uncertain"


def test_runtime_defaults_cache_hits_within_ttl(monkeypatch):
    calls = {"n": 0}

    def _fake_read(_resident_id):
        calls["n"] += 1
        return {"dataset_code": "muvim", "op_code": "OP-2"}

    monitor_route._RUNTIME_DEFAULTS_CACHE.clear()
    monkeypatch.setattr(monitor_route, "_read_runtime_defaults_db", _fake_read)
    monkeypatch.setattr(monitor_route, "_RUNTIME_DEFAULTS_TTL_S", 60.0)

    a = monitor_route._get_runtime_defaults_cached(1)
    b = monitor_route._get_runtime_defaults_cached(1)
    assert a["dataset_code"] == "muvim"
    assert b["op_code"] == "OP-2"
    assert calls["n"] == 1
