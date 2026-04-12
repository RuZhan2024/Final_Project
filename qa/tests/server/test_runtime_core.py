import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from applications.backend import core as core_mod
from applications.backend import deploy_runtime as dr
from applications.backend.online_alert import OnlineAlertTracker


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.current = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, _sql, _params=None):
        self.current = self.conn.responses.pop(0) if self.conn.responses else None

    def fetchone(self):
        if isinstance(self.current, dict):
            return self.current
        if isinstance(self.current, list) and self.current:
            return self.current[0]
        return {}

    def fetchall(self):
        if isinstance(self.current, list):
            return self.current
        return []


class _FakeConn:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.commits = 0

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.commits += 1


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def view(self, *_shape):
        return self

    def __getitem__(self, _idx):
        return self

    def item(self):
        if isinstance(self.value, np.ndarray):
            return float(np.asarray(self.value).reshape(-1)[0])
        return float(self.value)

    def to(self, **_kwargs):
        return self

    def unsqueeze(self, _dim):
        return self

    @property
    def shape(self):
        return np.asarray(self.value).shape


class _FakeTorch:
    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    @staticmethod
    def inference_mode():
        return _FakeTorch._Ctx()

    @staticmethod
    def from_numpy(arr):
        return _FakeTensor(np.asarray(arr, dtype=np.float32))

    float32 = np.float32


def test_core_jsonable_and_anonymize_xy():
    x = {"b": b"abc", "arr": [1, 2, {"k": 3}]}
    out = core_mod._jsonable(x)
    assert out["b"] == "abc"
    assert out["arr"][2]["k"] == 3

    xy = np.zeros((2, 33, 2), dtype=np.float32)
    xy[:, 23, :] = np.array([1.0, 2.0], dtype=np.float32)
    xy[:, 24, :] = np.array([3.0, 4.0], dtype=np.float32)
    anon = core_mod._anonymize_xy_inplace(xy.copy())
    assert np.allclose(anon[:, 23, :], np.array([[-1.0, -1.0], [-1.0, -1.0]], dtype=np.float32))


def test_core_derive_ops_params_from_yaml_modes(monkeypatch):
    specs = {
        "caucafall_tcn": SimpleNamespace(
            alert_cfg={"cooldown_s": 4, "ema_alpha": 0.2, "k": 2, "n": 3},
            ops={"OP-2": {"tau_low": 0.15, "tau_high": 0.8}},
        ),
        "caucafall_gcn": SimpleNamespace(
            alert_cfg={"cooldown_s": 5, "ema_alpha": 0.1, "k": 3, "n": 4},
            ops={"OP-2": {"tau_low": 0.2, "tau_high": 0.85}},
        ),
    }
    monkeypatch.setattr(core_mod, "_get_deploy_specs", lambda: specs)

    tcn = core_mod._derive_ops_params_from_yaml("caucafall", "TCN", "op2")
    gcn = core_mod._derive_ops_params_from_yaml("caucafall", "GCN", "op2")
    hyb = core_mod._derive_ops_params_from_yaml("caucafall", "HYBRID", "op2")

    assert tcn["ui"]["tau_high"] == 0.8
    assert gcn["ui"]["tau_high"] == 0.85
    assert hyb["ui"]["tau_high"] == 0.8
    assert hyb["ui"]["cooldown_s"] == 4.0


def test_deploy_runtime_discover_from_ops_yaml(tmp_path: Path):
    root = tmp_path
    (root / "configs" / "ops").mkdir(parents=True)
    ckpt = root / "outputs" / "muvim_gcn" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")

    yml = {
        "arch": "gcn",
        "ckpt": "outputs/muvim_gcn/best.pt",
        "feat_cfg": {"use_motion": True},
        "alert_cfg": {"k": 2, "n": 3},
        "ops": {
            "op1": {"tau_low": 0.1, "tau_high": 0.9},
            "op2": {"tau_low": 0.2, "tau_high": 0.8},
            "op3": {"tau_low": 0.3, "tau_high": 0.7},
        },
    }
    (root / "configs" / "ops" / "gcn_muvim.yaml").write_text(json.dumps(yml), encoding="utf-8")

    specs = dr._discover_from_ops_yaml(root)
    assert "muvim_gcn" in specs
    assert specs["muvim_gcn"].ops["OP-2"]["tau_high"] == 0.8


def test_deploy_runtime_alert_cfg_and_fuse():
    s = dr.DeploySpec(
        key="k",
        dataset="muvim",
        arch="gcn",
        ckpt="/tmp/none",
        feat_cfg={},
        model_cfg={},
        data_cfg={},
        alert_cfg={"tau_low": 0.4, "tau_high": 0.9},
        ops={"OP-2": {"tau_low": 0.2, "tau_high": 0.8}},
    )
    old = dr._SPECS
    try:
        dr._SPECS = {"k": s}
        cfg = dr.get_alert_cfg("k", "op2")
        tau_low, tau_high = dr.get_op_taus("k", "op2")
        assert cfg["tau_low"] == 0.2
        assert tau_high == 0.8
        assert dr.fuse_hybrid("not_fall", "not_fall") == "not_fall"
        assert dr.fuse_hybrid("fall", "uncertain") == "fall"
        assert dr.fuse_hybrid("uncertain", "not_fall") == "uncertain"
    finally:
        dr._SPECS = old


def test_get_pose_preprocess_cfg_prefers_checkpoint_over_spec(monkeypatch):
    fake_spec = dr.DeploySpec(
        key="muvim_tcn",
        dataset="muvim",
        arch="tcn",
        ckpt="/tmp/fake-best.pt",
        feat_cfg={},
        model_cfg={},
        data_cfg={"pose_preprocess": {"smooth_k": 3, "normalize": "none"}},
        alert_cfg={},
        ops={},
    )

    old_specs = dr._SPECS
    old_cache = dict(dr._POSE_PREPROCESS_CACHE)
    try:
        dr._SPECS = {"muvim_tcn": fake_spec}
        dr._POSE_PREPROCESS_CACHE.clear()

        calls = {"n": 0}

        def _fake_load_ckpt(_path, map_location="cpu"):
            calls["n"] += 1
            return {
                "data_cfg": {
                    "pose_preprocess": {
                        "smooth_k": 7,
                        "normalize": "shoulder",
                        "pelvis_fill": "forward",
                    }
                }
            }

        monkeypatch.setattr(dr, "_get_ml_runtime", lambda: {"load_ckpt": _fake_load_ckpt})

        cfg1 = dr.get_pose_preprocess_cfg("muvim_tcn")
        cfg2 = dr.get_pose_preprocess_cfg("muvim_tcn")

        assert cfg1["smooth_k"] == 7
        assert cfg1["normalize"] == "shoulder"
        assert cfg1["pelvis_fill"] == "forward"
        assert cfg2 == cfg1
        assert calls["n"] == 1
    finally:
        dr._SPECS = old_specs
        dr._POSE_PREPROCESS_CACHE.clear()
        dr._POSE_PREPROCESS_CACHE.update(old_cache)


def test_predict_spec_applies_mc_when_gate_allows(monkeypatch):
    fake_spec = dr.DeploySpec(
        key="muvim_tcn",
        dataset="muvim",
        arch="tcn",
        ckpt="/tmp/none",
        feat_cfg={},
        model_cfg={},
        data_cfg={},
        alert_cfg={},
        ops={"OP-2": {"tau_low": 0.2, "tau_high": 0.8}},
    )
    monkeypatch.setattr(dr, "get_specs", lambda: {"muvim_tcn": fake_spec})
    monkeypatch.setattr(dr, "_load_model_and_cfg", lambda _spec: {"model": object(), "device": "cpu", "feat_cfg": {}, "model_cfg": {}})
    monkeypatch.setattr(dr, "_prepare_features", lambda **_kwargs: {"kind": "tcn", "x_t": _FakeTensor([0.0])})
    monkeypatch.setattr(dr, "_get_optimized_infer_model", lambda **_kwargs: object())
    monkeypatch.setattr(dr, "_forward_prob", lambda _model, _prepared: _FakeTensor(0.55))
    monkeypatch.setattr(dr, "should_run_mc", lambda **_kwargs: (True, "boundary_window"))
    monkeypatch.setattr(
        dr,
        "_get_ml_runtime",
        lambda: {
            "torch": _FakeTorch(),
            "mc_predict_mu_sigma": lambda _model, forward_fn, M=16: (_FakeTensor(0.66), _FakeTensor(0.03)),
            "load_ckpt": lambda *_args, **_kwargs: {},
        },
    )

    out = dr.predict_spec(
        spec_key="muvim_tcn",
        joints_xy=np.zeros((48, 33, 2), dtype=np.float32),
        conf=np.ones((48, 33), dtype=np.float32),
        fps=25.0,
        target_T=48,
        op_code="OP-2",
        use_mc=True,
        mc_M=12,
    )

    assert out["p_det"] == 0.55
    assert out["mu"] == 0.66
    assert out["sigma"] == 0.03
    assert out["mc_applied"] is True
    assert out["mc_reason"] == "boundary_window"


def test_predict_spec_skips_mc_when_gate_blocks(monkeypatch):
    fake_spec = dr.DeploySpec(
        key="muvim_tcn",
        dataset="muvim",
        arch="tcn",
        ckpt="/tmp/none",
        feat_cfg={},
        model_cfg={},
        data_cfg={},
        alert_cfg={},
        ops={"OP-2": {"tau_low": 0.2, "tau_high": 0.8}},
    )
    monkeypatch.setattr(dr, "get_specs", lambda: {"muvim_tcn": fake_spec})
    monkeypatch.setattr(dr, "_load_model_and_cfg", lambda _spec: {"model": object(), "device": "cpu", "feat_cfg": {}, "model_cfg": {}})
    monkeypatch.setattr(dr, "_prepare_features", lambda **_kwargs: {"kind": "tcn", "x_t": _FakeTensor([0.0])})
    monkeypatch.setattr(dr, "_get_optimized_infer_model", lambda **_kwargs: object())
    monkeypatch.setattr(dr, "_forward_prob", lambda _model, _prepared: _FakeTensor(0.55))
    monkeypatch.setattr(dr, "should_run_mc", lambda **_kwargs: (False, "outside_boundary"))
    monkeypatch.setattr(
        dr,
        "_get_ml_runtime",
        lambda: {
            "torch": _FakeTorch(),
            "mc_predict_mu_sigma": lambda *_args, **_kwargs: (_FakeTensor(0.66), _FakeTensor(0.03)),
            "load_ckpt": lambda *_args, **_kwargs: {},
        },
    )

    out = dr.predict_spec(
        spec_key="muvim_tcn",
        joints_xy=np.zeros((48, 33, 2), dtype=np.float32),
        conf=np.ones((48, 33), dtype=np.float32),
        fps=25.0,
        target_T=48,
        op_code="OP-2",
        use_mc=True,
    )

    assert out["mu"] == 0.55
    assert out["sigma"] == 0.0
    assert out["mc_applied"] is False
    assert out["mc_reason"] == "outside_boundary"


def test_prepare_features_tcn_uses_runtime_builders(monkeypatch):
    seen = {"fps": None}

    def _build_canonical_input(**kwargs):
        seen["fps"] = kwargs["fps"]
        return np.zeros((4, 5, 2), dtype=np.float32), None

    monkeypatch.setattr(
        dr,
        "_get_ml_runtime",
        lambda: {
            "build_canonical_input": _build_canonical_input,
            "build_tcn_input": lambda xg, _feat: np.zeros((xg.shape[0], 10), dtype=np.float32),
            "split_gcn_two_stream": lambda xg, _feat: (xg, xg),
            "torch": _FakeTorch(),
        },
    )

    out = dr._prepare_features(
        spec=dr.DeploySpec("k", "caucafall", "tcn", "/tmp/x", {}, {}, {}, {}, {}),
        model=object(),
        device="cpu",
        feat_cfg={},
        model_cfg={"num_joints": 5},
        joints_xy=np.zeros((4, 5, 2), dtype=np.float32),
        conf=np.ones((4, 5), dtype=np.float32),
        fps=23.0,
    )

    assert out["kind"] == "tcn"
    assert seen["fps"] == 23.0
    assert out["x_t"].shape == (4, 10)


def test_prepare_features_gcn_two_stream_uses_runtime_splitter(monkeypatch):
    monkeypatch.setattr(
        dr,
        "_get_ml_runtime",
        lambda: {
            "build_canonical_input": lambda **_kwargs: (np.zeros((4, 5, 6), dtype=np.float32), None),
            "build_tcn_input": lambda xg, _feat: xg,
            "split_gcn_two_stream": lambda xg, _feat: (xg[..., :4], xg[..., 4:6]),
            "torch": _FakeTorch(),
        },
    )

    class _TwoStreamModel:
        two_stream = True

    out = dr._prepare_features(
        spec=dr.DeploySpec("k", "caucafall", "gcn", "/tmp/x", {}, {}, {}, {}, {}),
        model=_TwoStreamModel(),
        device="cpu",
        feat_cfg={},
        model_cfg={"num_joints": 5},
        joints_xy=np.zeros((4, 5, 2), dtype=np.float32),
        conf=np.ones((4, 5), dtype=np.float32),
        fps=23.0,
    )

    assert out["kind"] == "gcn_two_stream"
    assert out["xj_t"].shape == (4, 5, 4)
    assert out["xm_t"].shape == (4, 5, 2)


def test_align_joint_count_trims_and_pads():
    joints = np.zeros((2, 4, 2), dtype=np.float32)
    conf = np.ones((2, 4), dtype=np.float32)

    j_trim, c_trim = dr._align_joint_count(joints, conf, expected_v=3)
    assert j_trim.shape == (2, 3, 2)
    assert c_trim.shape == (2, 3)

    j_pad, c_pad = dr._align_joint_count(joints, conf, expected_v=6)
    assert j_pad.shape == (2, 6, 2)
    assert c_pad.shape == (2, 6)


def test_online_alert_tracker_transition_flow():
    trk = OnlineAlertTracker({"tau_low": 0.5, "tau_high": 0.8, "k": 2, "n": 3, "cooldown_s": 2.0, "ema_alpha": 0.0})
    r1 = trk.step(0.9, 0.0)
    r2 = trk.step(0.95, 0.1)
    assert r1.triage_state in {"uncertain", "not_fall", "fall"}
    assert r2.triage_state == "fall"
    assert r2.started_event is True
    r3 = trk.step(0.1, 0.2)
    assert r3.ended_event is True
    assert r3.cooldown_remaining_s > 0.0


def test_core_apply_settings_update_inmem():
    payload = core_mod.SettingsUpdatePayload(
        monitoring_enabled=True,
        alert_cooldown_sec=7,
        fall_threshold=90,
        active_model_code="gcn",
        active_dataset_code="LE2I",
        active_op_code="op-3",
        mc_M=12,
        mc_M_confirm=21,
    )
    core_mod.apply_settings_update_inmem(payload, resident_id=99)
    cfg = core_mod.get_inmem_settings(99)
    assert cfg["system"]["monitoring_enabled"] is True
    assert cfg["system"]["fall_threshold"] == 0.9
    assert cfg["system"]["active_model_code"] == "GCN"
    assert cfg["system"]["active_dataset_code"] == "le2i"
    assert cfg["system"]["active_op_code"] == "OP-3"
    assert cfg["deploy"]["mc"]["M"] == 12
    assert cfg["deploy"]["mc"]["M_confirm"] == 21


def test_core_db_helper_functions(monkeypatch):
    core_mod._TABLE_CACHE = None
    c1 = _FakeConn(responses=[[{"Tables_in_x": "events"}, {"Tables_in_x": "models"}]])
    tables = core_mod._list_tables(c1)
    assert "events" in tables and "models" in tables

    core_mod._COL_CACHE.clear()
    c2 = _FakeConn(responses=[[{"Field": "ts"}, {"Field": "score"}]])
    cols = core_mod._cols(c2, "events")
    assert "ts" in cols
    assert core_mod._has_col(c2, "events", "score") is True

    c3 = _FakeConn(responses=[{"id": 11}, {"ok": 1}])
    assert core_mod._one_resident_id(c3) == 11
    assert core_mod._resident_exists(c3, 11) is True

    c4 = _FakeConn(responses=[{"id": 5}, {"code": "GCN"}, {"id": 3, "model_id": 5}])
    assert core_mod._resolve_model_id(c4, "GCN") == 5
    assert core_mod._resolve_model_code(c4, 5) == "GCN"
    assert core_mod._resolve_op_id(c4, 5, 3) == 3

    monkeypatch.setattr(core_mod, "_cols", lambda _conn, _table: {"created_at", "p_fall"})
    assert core_mod._event_time_col(c4) == "created_at"
    assert core_mod._event_prob_col(c4) == "p_fall"


def test_core_clip_privacy_and_caregiver_table_paths(monkeypatch):
    c = _FakeConn(responses=[{"store_event_clips": 1, "anonymize_skeleton_data": 0}])
    monkeypatch.setattr(core_mod, "_ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(core_mod, "_detect_variants", lambda _c: {"settings": "v2", "events": "v1", "ops": "v1"})
    monkeypatch.setattr(core_mod, "_table_exists", lambda _c, t: t in {"system_settings"})

    store, anon = core_mod._read_clip_privacy_flags(c, resident_id=1)
    assert store is True
    assert anon is False

    c2 = _FakeConn()
    monkeypatch.setattr(core_mod, "_table_exists", lambda _c, t: False if t == "caregivers" else True)
    core_mod._ensure_caregivers_table(c2)
    assert c2.commits >= 1


def test_core_system_settings_schema_and_variants(monkeypatch):
    c = _FakeConn(responses=[[{"Tables_in_x": "system_settings"}], [{"Field": "fall_threshold"}]])
    core_mod._TABLE_CACHE = None
    core_mod._COL_CACHE.clear()
    core_mod._ensure_system_settings_schema(c)
    assert c.commits >= 1

    monkeypatch.setattr(
        core_mod,
        "_cols",
        lambda _conn, table: {
            "system_settings": {"active_model_id"},
            "events": {"event_time"},
            "operating_points": {"model_id"},
        }[table],
    )
    v = core_mod._detect_variants(c)
    assert v == {"settings": "v2", "events": "v2", "ops": "v2"}


def test_core_resolvers_fallback_branches(monkeypatch):
    c = _FakeConn(responses=[{}, {"id": 7}])
    assert core_mod._resolve_model_id(c, "GCN") == 7

    c2 = _FakeConn(responses=[{"id": 4, "model_id": 123}])
    assert core_mod._resolve_op_id(c2, model_id=999, op_id=4) is None

    monkeypatch.setattr(core_mod, "_cols", lambda _conn, _table: {"something_else"})
    assert core_mod._event_prob_col(c2) is None


def test_deploy_runtime_discover_from_reports(tmp_path: Path):
    root = tmp_path
    reports = root / "outputs" / "reports"
    reports.mkdir(parents=True)
    ckpt = root / "outputs" / "le2i_tcn" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")

    rep = {
        "arch": "tcn",
        "ckpt": "outputs/le2i_tcn/best.pt",
        "feat_cfg": {"x": 1},
        "model_cfg": {"h": 2},
        "data_cfg": {"fps_default": 25},
        "ops_eval": {
            "op1": {"alert_cfg": {"tau_low": 0.1, "tau_high": 0.9}},
            "op2": {"alert_cfg": {"tau_low": 0.2, "tau_high": 0.8}},
            "op3": {"alert_cfg": {"tau_low": 0.3, "tau_high": 0.7}},
        },
    }
    (reports / "le2i_tcn.json").write_text(json.dumps(rep), encoding="utf-8")

    specs = dr._discover_from_reports(root)
    assert "le2i_tcn" in specs
    assert specs["le2i_tcn"].ops["OP-2"]["tau_high"] == 0.8
