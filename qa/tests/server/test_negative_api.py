from contextlib import contextmanager

from fastapi.testclient import TestClient

from applications.backend.main import app
from applications.backend.routes import events as events_route
from applications.backend.routes import monitor as monitor_route


def test_monitor_predict_window_rejects_missing_xy_payload():
    client = TestClient(app)
    resp = client.post("/api/monitor/predict_window", json={"mode": "dual"})
    assert resp.status_code == 400
    assert "payload must include raw_*" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_unknown_dataset_specs(monkeypatch):
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", lambda: {})

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "dataset_code": "muvim",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
        },
    )
    assert resp.status_code == 404
    assert "No deploy specs found" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_invalid_target_t():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "dataset_code": "muvim",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": "bad-value",
        },
    )
    assert resp.status_code == 400
    assert "target_T must be an integer" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_invalid_mc_m():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "dataset_code": "muvim",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
            "mc_M": 0,
        },
    )
    assert resp.status_code == 400
    assert "mc_M must be >= 1" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_invalid_mc_sigma_tol():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "dataset_code": "muvim",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
            "mc_sigma_tol": "bad-value",
        },
    )
    assert resp.status_code == 400
    assert "mc_sigma_tol must be a positive number" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_invalid_mc_se_tol():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "dataset_code": "muvim",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
            "mc_se_tol": "bad-value",
        },
    )
    assert resp.status_code == 400
    assert "mc_se_tol must be a positive number" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_oversized_raw_stream():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "raw_t_ms": list(range(5000)),
            "raw_xy_flat": [0.0, 0.0] * 5000,
            "raw_joints": 1,
            "target_T": 2,
        },
    )
    assert resp.status_code == 413
    assert "raw_t_ms too long" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_invalid_raw_joints():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "raw_t_ms": [0.0, 40.0],
            "raw_xy_flat": [0.0, 0.0, 0.1, 0.1],
            "raw_joints": 99,
            "target_T": 2,
        },
    )
    assert resp.status_code == 400
    assert "raw_joints must be <= 64" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_raw_xy_flat_length_mismatch():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "raw_t_ms": [0.0, 40.0],
            "raw_xy_flat": [0.0, 0.0, 0.1],  # expected 4 values for 2 frames x 1 joint x 2
            "raw_joints": 1,
            "target_T": 2,
        },
    )
    assert resp.status_code == 400
    assert "raw_xy_flat length mismatch" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_raw_conf_flat_length_mismatch():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "raw_t_ms": [0.0, 40.0],
            "raw_xy_flat": [0.0, 0.0, 0.1, 0.1],
            "raw_conf_flat": [1.0],  # expected 2 values for 2 frames x 1 joint
            "raw_joints": 1,
            "target_T": 2,
        },
    )
    assert resp.status_code == 400
    assert "raw_conf_flat length mismatch" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_non_numeric_raw_xy_flat():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "raw_t_ms": [0.0, 40.0],
            "raw_xy_flat": [0.0, "bad", 0.1, 0.1],
            "raw_joints": 1,
            "target_T": 2,
        },
    )
    assert resp.status_code == 400
    assert "raw_xy_flat must be a flat numeric array" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_non_numeric_raw_conf_flat():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "raw_t_ms": [0.0, 40.0],
            "raw_xy_flat": [0.0, 0.0, 0.1, 0.1],
            "raw_conf_flat": ["bad", 1.0],
            "raw_joints": 1,
            "target_T": 2,
        },
    )
    assert resp.status_code == 400
    assert "raw_conf_flat must be a flat numeric array" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_non_numeric_raw_t_ms():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "raw_t_ms": [0.0, "bad"],
            "raw_xy_flat": [0.0, 0.0, 0.1, 0.1],
            "raw_joints": 1,
            "target_T": 2,
        },
    )
    assert resp.status_code == 400
    assert "raw_t_ms must be a numeric array" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_non_monotonic_raw_t_ms():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "raw_t_ms": [0.0, 40.0, 40.0],
            "raw_xy_flat": [0.0, 0.0, 0.1, 0.1, 0.2, 0.2],
            "raw_joints": 1,
            "target_T": 2,
        },
    )
    assert resp.status_code == 400
    assert "raw_t_ms must be strictly increasing" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_invalid_direct_xy_shape():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "mode": "tcn",
            "dataset_code": "muvim",
            "xy": [[0.0, 0.0], [0.1, 0.1]],  # not [T,J,2]
            "conf": [[1.0], [1.0]],
            "target_T": 2,
        },
    )
    assert resp.status_code == 400
    assert "xy must be shaped [T,J,2]" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_direct_conf_shape_mismatch():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "mode": "tcn",
            "dataset_code": "muvim",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],  # T=2, J=1
            "conf": [[1.0, 1.0], [1.0, 1.0]],    # T=2, J=2 mismatch
            "target_T": 2,
        },
    )
    assert resp.status_code == 400
    assert "conf shape mismatch for xy" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_direct_xy_target_t_mismatch():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "mode": "tcn",
            "dataset_code": "muvim",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]], [[0.2, 0.2]]],  # T=3
            "conf": [[1.0], [1.0], [1.0]],
            "target_T": 2,
        },
    )
    assert resp.status_code == 400
    assert "xy time length mismatch" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_non_numeric_direct_xy():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "mode": "tcn",
            "dataset_code": "muvim",
            "xy": [[[0.0, 0.0]], [["bad", 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
        },
    )
    assert resp.status_code == 400
    assert "xy must be a numeric array" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_non_numeric_direct_conf():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "mode": "tcn",
            "dataset_code": "muvim",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [["bad"], [1.0]],
            "target_T": 2,
        },
    )
    assert resp.status_code == 400
    assert "conf must be a numeric array" in resp.json().get("detail", "")


def test_events_list_rejects_invalid_model_query():
    client = TestClient(app)
    resp = client.get("/api/events?model=bad-model")
    assert resp.status_code == 400
    assert "model must be one of" in resp.json().get("detail", "")


def test_events_list_rejects_invalid_status_query():
    client = TestClient(app)
    resp = client.get("/api/events?status=totally_invalid")
    assert resp.status_code == 400
    assert "status must be one of" in resp.json().get("detail", "")


def test_operating_points_rejects_invalid_model_code():
    client = TestClient(app)
    resp = client.get("/api/operating_points?model_code=BAD")
    assert resp.status_code == 400
    assert "model_code must be one of" in resp.json().get("detail", "")


def test_events_list_accepts_known_values_and_reaches_db_path(monkeypatch):
    # Guardrail test to ensure validation doesn't reject valid inputs.
    @contextmanager
    def _no_db():
        yield None

    monkeypatch.setattr(events_route, "get_conn_optional", _no_db)
    client = TestClient(app)
    resp = client.get("/api/events?model=GCN&status=pending_review")
    # DB is unavailable in this mocked path, but validation should pass and endpoint should return 200 fallback shape.
    assert resp.status_code == 200
