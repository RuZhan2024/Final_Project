from contextlib import contextmanager

from fastapi.testclient import TestClient

from applications.backend.main import app
from applications.backend.routes import events as events_route
from applications.backend.routes import monitor as monitor_route


def test_monitor_predict_window_rejects_retired_dual_mode():
    client = TestClient(app)
    resp = client.post("/api/monitor/predict_window", json={"mode": "dual"})

    assert resp.status_code == 400
    assert "mode='tcn'" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_missing_pose_payload():
    client = TestClient(app)
    resp = client.post("/api/monitor/predict_window", json={"mode": "tcn"})

    assert resp.status_code == 400
    assert "payload must include raw_*" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_unknown_deploy_spec(monkeypatch):
    monkeypatch.setattr(monitor_route, "_get_deploy_specs", lambda: {})

    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "dataset_code": "caucafall",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
        },
    )

    assert resp.status_code == 404
    assert "No TCN deploy spec found" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_invalid_target_t_type():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "dataset_code": "caucafall",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": "bad-value",
        },
    )

    assert resp.status_code == 422


def test_monitor_predict_window_rejects_invalid_direct_xy_shape():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "mode": "tcn",
            "dataset_code": "caucafall",
            "xy": [[0.0, 0.0], [0.1, 0.1]],
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
            "dataset_code": "caucafall",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]]],
            "conf": [[1.0, 1.0], [1.0, 1.0]],
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
            "dataset_code": "caucafall",
            "xy": [[[0.0, 0.0]], [[0.1, 0.1]], [[0.2, 0.2]]],
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
            "dataset_code": "caucafall",
            "xy": [[[0.0, 0.0]], [["bad", 0.1]]],
            "conf": [[1.0], [1.0]],
            "target_T": 2,
        },
    )

    assert resp.status_code == 400
    assert "xy must be numeric" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_oversized_raw_stream():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "raw_t_ms": list(range(5000)),
            "raw_xy": [[[0.0, 0.0]]] * 5000,
            "raw_conf": [[1.0]] * 5000,
            "target_T": 2,
        },
    )

    assert resp.status_code == 413
    assert "raw_t_ms too long" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_raw_t_ms_xy_length_mismatch():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "raw_t_ms": [0.0, 40.0],
            "raw_xy": [[[0.0, 0.0]]],
            "raw_conf": [[1.0]],
            "target_T": 2,
        },
    )

    assert resp.status_code == 400
    assert "raw_t_ms and raw_xy time dimensions must match" in resp.json().get("detail", "")


def test_monitor_predict_window_rejects_non_monotonic_raw_t_ms():
    client = TestClient(app)
    resp = client.post(
        "/api/monitor/predict_window",
        json={
            "raw_t_ms": [0.0, 40.0, 40.0],
            "raw_xy": [[[0.0, 0.0]], [[0.1, 0.1]], [[0.2, 0.2]]],
            "raw_conf": [[1.0], [1.0], [1.0]],
            "target_T": 2,
        },
    )

    assert resp.status_code == 400
    assert "raw_t_ms must be strictly increasing" in resp.json().get("detail", "")


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
    @contextmanager
    def _no_db():
        yield None

    monkeypatch.setattr(events_route, "get_conn_optional", _no_db)
    client = TestClient(app)
    resp = client.get("/api/events?model=CTR_GCN&status=pending_review")

    assert resp.status_code == 200
    assert resp.json()["db_available"] is False
