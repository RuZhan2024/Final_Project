from __future__ import annotations

from fastapi.testclient import TestClient

from applications.backend.app import app
from applications.backend.deploy_runtime import discover_specs, _discover_from_ops_yaml
import applications.backend.main as server_main
from applications.backend.routes import monitor as monitor_routes
from fall_detection.preprocessing.pose_resample import resample_pose_window


def test_discover_specs_reads_ops_yaml_schema() -> None:
    specs = discover_specs()
    assert isinstance(specs, dict)
    for k in specs.keys():
        assert isinstance(k, str)


def test_cors_allowed_origins_defaults_and_env(monkeypatch) -> None:
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    defaults = server_main._compute_allowed_origins()
    assert "http://localhost:3000" in defaults

    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,https://example.com")
    custom = server_main._compute_allowed_origins()
    assert custom == ["http://localhost:3000", "https://example.com"]


def test_discover_from_ops_yaml_supports_nested_model_schema_and_relative_ckpt(tmp_path) -> None:
    root = tmp_path
    ops_dir = root / "configs" / "ops"
    ckpt = root / "outputs" / "demo_tcn" / "best.pt"
    ops_dir.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"stub")

    yaml_path = ops_dir / "tcn_demo.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "ops:",
                "  OP2:",
                "    tau_low: 0.4",
                "    tau_high: 0.6",
                "alert_cfg:",
                "  ema_alpha: 0.2",
                "model:",
                "  arch: \"tcn\"",
                "  ckpt: \"../../outputs/demo_tcn/best.pt\"",
                "  feat_cfg:",
                "    center: \"pelvis\"",
            ]
        ),
        encoding="utf-8",
    )

    specs = _discover_from_ops_yaml(root)
    assert "demo_tcn" in specs
    spec = specs["demo_tcn"]
    assert spec.arch == "tcn"
    assert spec.dataset == "demo"
    assert spec.ckpt == str(ckpt.resolve())
    assert spec.feat_cfg.get("center") == "pelvis"


def test_monitor_mode_specific_missing_spec_errors(monkeypatch) -> None:
    monkeypatch.setattr(monitor_routes, "_get_deploy_specs", lambda: {})
    c = TestClient(app)
    base = {
        "session_id": "audit",
        "dataset_code": "no_ds",
        "raw_t_ms": [0, 40],
        "raw_xy": [[[0.0, 0.0]], [[0.0, 0.0]]],
        "window_end_t_ms": 40,
    }

    r_tcn = c.post("/api/monitor/predict_window", json={**base, "mode": "tcn"})
    assert r_tcn.status_code == 404
    assert "No TCN deploy spec found" in r_tcn.json().get("detail", "")

    r_gcn = c.post("/api/monitor/predict_window", json={**base, "mode": "gcn"})
    assert r_gcn.status_code == 404
    assert "No GCN deploy spec found" in r_gcn.json().get("detail", "")

    r_dual = c.post("/api/monitor/predict_window", json={**base, "mode": "dual"})
    assert r_dual.status_code == 404
    assert "No deploy specs found" in r_dual.json().get("detail", "")


def test_notifications_test_endpoint_exists() -> None:
    c = TestClient(app)
    r = c.post("/api/notifications/test", json={"message": "smoke"})
    assert r.status_code == 200
    payload = r.json()
    assert payload.get("ok") is True
    assert payload.get("accepted") is True
    rv1 = c.post("/api/v1/notifications/test", json={"message": "smoke"})
    assert rv1.status_code == 200
    assert rv1.json().get("ok") is True


def test_event_status_endpoint_exists_without_db() -> None:
    c = TestClient(app)
    r = c.put("/api/events/123/status", json={"status": "confirmed_fall"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert body.get("persisted") is False
    rv1 = c.put("/api/v1/events/123/status", json={"status": "confirmed_fall"})
    assert rv1.status_code == 200
    assert rv1.json().get("ok") is True


def test_events_summary_includes_today_shape() -> None:
    c = TestClient(app)
    r = c.get("/api/events/summary?resident_id=1")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body.get("today"), dict)
    today = body["today"]
    for k in ("falls", "pending", "false_alarms"):
        assert k in today


def test_v1_settings_and_events_aliases() -> None:
    c = TestClient(app)
    rg = c.get("/api/v1/settings")
    assert rg.status_code == 200
    assert isinstance(rg.json().get("system"), dict)

    rp = c.put("/api/v1/settings", json={"monitoring_enabled": True})
    assert rp.status_code == 200
    assert rp.json().get("ok") is True

    re = c.get("/api/v1/events?resident_id=1&limit=5")
    assert re.status_code == 200
    assert isinstance(re.json().get("events"), list)

    rs = c.get("/api/v1/events/summary?resident_id=1")
    assert rs.status_code == 200
    assert isinstance(rs.json().get("today"), dict)

    rc = c.get("/api/v1/caregivers?resident_id=1")
    assert rc.status_code == 200
    assert isinstance(rc.json().get("caregivers"), list)

    rop = c.get("/api/v1/operating_points?model_code=TCN")
    assert rop.status_code == 200
    assert isinstance(rop.json().get("operating_points"), list)

    rsum = c.get("/api/v1/summary")
    assert rsum.status_code == 200
    assert isinstance(rsum.json().get("today"), dict)

    rds = c.get("/api/v1/dashboard/summary")
    assert rds.status_code == 200
    assert isinstance(rds.json().get("today"), dict)

    rtf = c.post("/api/v1/events/test_fall")
    assert rtf.status_code == 200
    assert "ok" in rtf.json()

    rclip = c.post(
        "/api/v1/events/123/skeleton_clip",
        json={"resident_id": 1, "t_ms": [0.0, 40.0], "xy": [[[0.0, 0.0]], [[0.1, 0.1]]], "conf": [[1.0], [1.0]]},
    )
    # No DB in this environment; alias reachability is what we verify.
    assert rclip.status_code in {503, 404}


def test_deploy_modes_endpoint_exposes_yaml_profile() -> None:
    c = TestClient(app)
    r = c.get("/api/deploy/modes")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body.get("deploy_modes"), dict)
    r_v1 = c.get("/api/v1/deploy/modes")
    assert r_v1.status_code == 200
    body_v1 = r_v1.json()
    assert isinstance(body_v1.get("deploy_modes"), dict)


def test_v1_spec_aliases() -> None:
    c = TestClient(app)
    r1 = c.get("/api/v1/spec")
    r2 = c.get("/api/v1/deploy/specs")
    r3 = c.get("/api/v1/models/summary")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 200


def test_v1_health_alias() -> None:
    c = TestClient(app)
    r = c.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_v1_monitor_reset_session_alias() -> None:
    c = TestClient(app)
    r = c.post("/api/v1/monitor/reset_session?session_id=smoke")
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("session_id") == "smoke"


def test_server_resample_parity_with_shared_helper() -> None:
    raw_t = [0.0, 40.0, 80.0, 120.0]
    raw_xy = [
        [[0.0, 0.0], [1.0, 1.0]],
        [[0.1, 0.2], [1.1, 1.2]],
        [[0.2, 0.4], [1.2, 1.4]],
        [[0.3, 0.6], [1.3, 1.6]],
    ]
    raw_conf = [
        [1.0, 0.9],
        [1.0, 0.9],
        [1.0, 0.9],
        [1.0, 0.9],
    ]
    kwargs = {
        "raw_t_ms": raw_t,
        "raw_xy": raw_xy,
        "raw_conf": raw_conf,
        "target_fps": 25.0,
        "target_T": 4,
        "window_end_t_ms": 120.0,
    }
    xy_s, conf_s, start_s, end_s, _fps_s = monitor_routes._resample_pose_window(**kwargs)
    xy_p, conf_p, start_p, end_p, _fps_p = resample_pose_window(**kwargs)

    assert len(xy_s) == len(xy_p) == 4
    assert len(conf_s) == len(conf_p) == 4
    assert abs(float(start_s) - float(start_p)) < 1e-4
    assert abs(float(end_s) - float(end_p)) < 1e-4


def test_predict_window_contract_with_mocked_runtime(monkeypatch) -> None:
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
            "p_det": 0.7,
            "mu": 0.7,
            "sigma": 0.0,
            "tau_low": 0.4,
            "tau_high": 0.6,
            "ops": {"OP-2": {"tau_low": 0.4, "tau_high": 0.6}},
            "alert_cfg": {"tau_low": 0.4, "tau_high": 0.6, "ema_alpha": 0.2, "k": 2, "n": 3, "cooldown_s": 30.0},
        },
    )

    c = TestClient(app)
    payload = {
        "session_id": "contract-smoke",
        "mode": "tcn",
        "dataset_code": "le2i",
        "op_code": "OP-2",
        "target_T": 48,
        "raw_t_ms": [0, 40],
        "raw_xy": [[[0.1, 0.2]], [[0.1, 0.2]]],
        "raw_conf": [[1.0], [1.0]],
        "window_end_t_ms": 40,
    }
    r = c.post("/api/monitor/predict_window", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body.get("dataset_code") == "le2i"
    assert body.get("op_code") == "OP-2"
    assert "triage_state" in body

    rv1 = c.post("/api/v1/monitor/predict_window", json=payload)
    assert rv1.status_code == 200


def test_predict_window_dual_falls_back_to_single_when_only_one_spec(monkeypatch) -> None:
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
            "p_det": 0.7,
            "mu": 0.7,
            "sigma": 0.0,
            "tau_low": 0.4,
            "tau_high": 0.6,
            "ops": {"OP-2": {"tau_low": 0.4, "tau_high": 0.6}},
            "alert_cfg": {"tau_low": 0.4, "tau_high": 0.6, "ema_alpha": 0.2, "k": 2, "n": 3, "cooldown_s": 30.0},
        },
    )

    c = TestClient(app)
    payload = {
        "session_id": "contract-smoke-fallback",
        "mode": "dual",
        "dataset_code": "le2i",
        "op_code": "OP-2",
        "target_T": 48,
        "raw_t_ms": [0, 40],
        "raw_xy": [[[0.1, 0.2]], [[0.1, 0.2]]],
        "raw_conf": [[1.0], [1.0]],
        "window_end_t_ms": 40,
    }
    r = c.post("/api/monitor/predict_window", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body.get("requested_mode") == "dual"
    assert body.get("effective_mode") == "tcn"


def test_predict_window_resolves_dataset_variant_spec_key(monkeypatch) -> None:
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

    variant_key = "le2i_hneg_pack_tsm_tcn"
    monkeypatch.setattr(
        monitor_routes,
        "_get_deploy_specs",
        lambda: {variant_key: _Spec(variant_key, "le2i_hneg_pack_tsm", "tcn")},
    )
    monkeypatch.setattr(
        monitor_routes,
        "_predict_spec",
        lambda **kwargs: {
            "spec_key": kwargs["spec_key"],
            "dataset": "le2i_hneg_pack_tsm",
            "arch": "tcn",
            "p_det": 0.7,
            "mu": 0.7,
            "sigma": 0.0,
            "tau_low": 0.4,
            "tau_high": 0.6,
            "ops": {"OP-2": {"tau_low": 0.4, "tau_high": 0.6}},
            "alert_cfg": {"tau_low": 0.4, "tau_high": 0.6, "ema_alpha": 0.2, "k": 2, "n": 3, "cooldown_s": 30.0},
        },
    )

    c = TestClient(app)
    payload = {
        "session_id": "contract-smoke-variant",
        "mode": "tcn",
        "dataset_code": "le2i",
        "op_code": "OP-2",
        "target_T": 48,
        "raw_t_ms": [0, 40],
        "raw_xy": [[[0.1, 0.2]], [[0.1, 0.2]]],
        "raw_conf": [[1.0], [1.0]],
        "window_end_t_ms": 40,
    }
    r = c.post("/api/monitor/predict_window", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body.get("effective_mode") == "tcn"
    assert body.get("models", {}).get("tcn", {}).get("spec_key") == variant_key
