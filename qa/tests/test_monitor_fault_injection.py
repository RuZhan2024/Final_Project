from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from applications.backend.app import app
from applications.backend.routes import monitor as monitor_routes


def _install_mock_runtime(monkeypatch) -> None:
    class _Spec:
        def __init__(self, key: str, dataset: str, arch: str) -> None:
            self.key = key
            self.dataset = dataset
            self.arch = arch
            self.ckpt = "unused.pt"
            self.feat_cfg = {}
            self.model_cfg = {}
            self.data_cfg = {}
            self.alert_cfg = {"tau_low": 0.4, "tau_high": 0.6, "ema_alpha": 0.2, "k": 2, "n": 3, "cooldown_s": 30.0}
            self.ops = {"OP-2": {"tau_low": 0.4, "tau_high": 0.6}}

    monkeypatch.setattr(
        monitor_routes,
        "_get_deploy_specs",
        lambda: {"le2i_tcn": _Spec("le2i_tcn", "le2i", "tcn")},
    )
    monkeypatch.setattr(
        monitor_routes,
        "_predict_spec",
        lambda **_kwargs: {
            "spec_key": "le2i_tcn",
            "dataset": "le2i",
            "arch": "tcn",
            "p_det": 0.6,
            "mu": 0.6,
            "sigma": 0.0,
            "tau_low": 0.4,
            "tau_high": 0.6,
            "ops": {"OP-2": {"tau_low": 0.4, "tau_high": 0.6}},
            "alert_cfg": {"tau_low": 0.4, "tau_high": 0.6, "ema_alpha": 0.2, "k": 2, "n": 3, "cooldown_s": 30.0},
        },
    )


def test_predict_window_handles_non_monotonic_timestamps(monkeypatch) -> None:
    _install_mock_runtime(monkeypatch)
    c = TestClient(app)
    payload = {
        "session_id": "fault-nonmono",
        "mode": "tcn",
        "dataset_code": "le2i",
        "op_code": "OP-2",
        "target_T": 48,
        "raw_t_ms": [0, 40, 20, 80],
        "raw_xy": [[[0.1, 0.2]], [[0.1, 0.2]], [[0.1, 0.2]], [[0.1, 0.2]]],
        "raw_conf": [[1.0], [1.0], [1.0], [1.0]],
        "window_end_t_ms": 80,
    }
    r = c.post("/api/monitor/predict_window", json=payload)
    assert r.status_code == 200
    assert "triage_state" in r.json()


def test_predict_window_handles_conf_missing_and_sparse_xy(monkeypatch) -> None:
    _install_mock_runtime(monkeypatch)
    c = TestClient(app)
    payload = {
        "session_id": "fault-sparse",
        "mode": "tcn",
        "dataset_code": "le2i",
        "op_code": "OP-2",
        "target_T": 48,
        "raw_t_ms": [0, 40, 80],
        "raw_xy": [[[0.1, 0.2]], [[0.2, 0.3]], [[0.3, 0.4]]],
        "window_end_t_ms": 80,
    }
    r = c.post("/api/monitor/predict_window", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body.get("dataset_code") == "le2i"
    assert body.get("effective_mode") == "tcn"


def test_predict_window_rejects_empty_raw_window(monkeypatch) -> None:
    _install_mock_runtime(monkeypatch)
    c = TestClient(app)
    payload = {
        "session_id": "fault-empty",
        "mode": "tcn",
        "dataset_code": "le2i",
        "op_code": "OP-2",
        "target_T": 48,
        "raw_t_ms": [],
        "raw_xy": [],
        "window_end_t_ms": 0,
    }
    r = c.post("/api/monitor/predict_window", json=payload)
    assert r.status_code == 400
    assert "raw_*" in r.json().get("detail", "")


def test_predict_window_logs_db_default_read_failure(monkeypatch, caplog) -> None:
    _install_mock_runtime(monkeypatch)

    def _boom_get_conn():
        raise RuntimeError("db offline")

    monkeypatch.setattr(monitor_routes, "get_conn", _boom_get_conn)
    c = TestClient(app)

    payload = {
        "session_id": "fault-db-log",
        "mode": "tcn",
        "dataset_code": "le2i",
        "op_code": "OP-2",
        "target_T": 48,
        "xy": [[[0.1, 0.2]] for _ in range(48)],
        "conf": [[1.0] for _ in range(48)],
    }
    with caplog.at_level(logging.WARNING, logger="server.routes.monitor"):
        r = c.post("/api/monitor/predict_window", json=payload)

    assert r.status_code == 200
    assert "monitor.predict_window: failed to load DB defaults" in caplog.text
    assert "session_id=fault-db-log" in caplog.text
