import json
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from server import core as core_mod
from server import deploy_runtime as dr
from server.online_alert import OnlineAlertTracker


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


def test_core_jsonable_and_anonymize_xy():
    x = {
        "b": b"abc",
        "arr": [1, 2, {"k": 3}],
    }
    out = core_mod._jsonable(x)
    assert out["b"] == "abc"
    assert out["arr"][2]["k"] == 3

    xy = np.zeros((2, 33, 2), dtype=np.float32)
    xy[:, 23, :] = np.array([1.0, 2.0], dtype=np.float32)
    xy[:, 24, :] = np.array([3.0, 4.0], dtype=np.float32)
    anon = core_mod._anonymize_xy_inplace(xy.copy())
    # pelvis center is [2,3], so joint 23 becomes [-1,-1]
    assert np.allclose(anon[:, 23, :], np.array([[-1.0, -1.0], [-1.0, -1.0]], dtype=np.float32))


def test_core_derive_ops_params_from_yaml_modes(monkeypatch):
    specs = {
        "muvim_tcn": SimpleNamespace(
            alert_cfg={"cooldown_s": 4, "ema_alpha": 0.2, "k": 2, "n": 3},
            ops={"OP-2": {"tau_low": 0.15, "tau_high": 0.8}},
        ),
        "muvim_gcn": SimpleNamespace(
            alert_cfg={"cooldown_s": 5, "ema_alpha": 0.1, "k": 3, "n": 4},
            ops={"OP-2": {"tau_low": 0.2, "tau_high": 0.85}},
        ),
    }
    monkeypatch.setattr(core_mod, "_get_deploy_specs", lambda: specs)

    tcn = core_mod._derive_ops_params_from_yaml("muvim", "TCN", "op2")
    gcn = core_mod._derive_ops_params_from_yaml("muvim", "GCN", "op2")
    hyb = core_mod._derive_ops_params_from_yaml("muvim", "HYBRID", "op2")

    assert tcn["ui"]["tau_high"] == 0.8
    assert gcn["ui"]["tau_high"] == 0.85
    assert hyb["ui"]["tau_high"] == 0.8
    assert hyb["ui"]["cooldown_s"] == 5.0


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
    spec = specs["muvim_gcn"]
    assert spec.arch == "gcn"
    assert spec.ops["OP-2"]["tau_high"] == 0.8


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


def test_predict_spec_mc_success_skips_extra_deterministic_forward(monkeypatch):
    class _TensorLike:
        def __init__(self, v):
            self.v = float(v)

        def cpu(self):
            return self

        def view(self, *_shape):
            return self

        def __getitem__(self, _idx):
            return self

        def item(self):
            return self.v

    class _FakeTorch:
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        @staticmethod
        def inference_mode():
            return _FakeTorch._Ctx()

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
    monkeypatch.setattr(dr, "_torch", lambda: _FakeTorch())
    monkeypatch.setattr(
        dr,
        "_load_model_and_cfg",
        lambda _spec: {"model": object(), "device": SimpleNamespace(type="cpu"), "feat_cfg": SimpleNamespace(), "is_two_stream": False},
    )
    monkeypatch.setattr("core.features.build_canonical_input", lambda **_kwargs: (np.zeros((48, 33, 2), dtype=np.float32), None))

    called = {"det": 0}

    def _forward():
        called["det"] += 1
        raise AssertionError("deterministic forward should not run before MC success path")

    monkeypatch.setattr(dr, "_make_forward_fn", lambda **_kwargs: _forward)

    fake_unc = types.SimpleNamespace(
        mc_predict_mu_sigma=lambda _model, forward_fn, M=10: (_TensorLike(0.73), _TensorLike(0.04))
    )
    monkeypatch.setitem(__import__("sys").modules, "core.uncertainty", fake_unc)

    out = dr.predict_spec(
        spec_key="muvim_tcn",
        joints_xy=np.zeros((48, 33, 2), dtype=np.float32),
        conf=np.ones((48, 33), dtype=np.float32),
        fps=25.0,
        target_T=48,
        op_code="OP-2",
        use_mc=True,
        mc_M=10,
    )
    assert called["det"] == 0
    assert out["p_det"] == 0.73
    assert out["mu"] == 0.73
    assert out["sigma"] == 0.04
    assert out["mc_n_used"] == 10


def test_predict_spec_passes_mc_sigma_and_se_tol_to_uncertainty(monkeypatch):
    class _TensorLike:
        def __init__(self, v):
            self.v = float(v)

        def cpu(self):
            return self

        def view(self, *_shape):
            return self

        def __getitem__(self, _idx):
            return self

        def item(self):
            return self.v

    class _FakeTorch:
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        @staticmethod
        def inference_mode():
            return _FakeTorch._Ctx()

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
    monkeypatch.setattr(dr, "_torch", lambda: _FakeTorch())
    monkeypatch.setattr(
        dr,
        "_load_model_and_cfg",
        lambda _spec: {"model": object(), "device": SimpleNamespace(type="cpu"), "feat_cfg": SimpleNamespace(), "is_two_stream": False},
    )
    monkeypatch.setattr("core.features.build_canonical_input", lambda **_kwargs: (np.zeros((48, 33, 2), dtype=np.float32), None))
    monkeypatch.setattr(dr, "_make_forward_fn", lambda **_kwargs: (lambda: _TensorLike(0.55)))

    seen = {}

    def _mc(_model, forward_fn, **kwargs):
        seen.update(kwargs)
        return _TensorLike(0.66), _TensorLike(0.03), 7

    monkeypatch.setitem(__import__("sys").modules, "core.uncertainty", types.SimpleNamespace(mc_predict_mu_sigma=_mc))

    out = dr.predict_spec(
        spec_key="muvim_tcn",
        joints_xy=np.zeros((48, 33, 2), dtype=np.float32),
        conf=np.ones((48, 33), dtype=np.float32),
        fps=25.0,
        target_T=48,
        op_code="OP-2",
        use_mc=True,
        mc_M=12,
        mc_sigma_tol=0.025,
        mc_se_tol=0.01,
    )

    assert out["mu"] == 0.66
    assert out["mc_n_used"] == 7
    assert seen.get("M") == 12
    assert seen.get("max_sigma_for_early_stop") == 0.025
    assert seen.get("max_se_for_early_stop") == 0.01
    assert seen.get("return_n_used") is True


def test_make_forward_fn_tcn_raises_on_in_ch_mismatch(monkeypatch):
    monkeypatch.setattr("core.features.build_tcn_input", lambda _xg, _feat: np.zeros((48, 64), dtype=np.float32))

    with __import__("pytest").raises(RuntimeError, match="TCN input dimension mismatch"):
        dr._make_forward_fn(
            torch=object(),
            model=object(),
            device=SimpleNamespace(type="cpu"),
            arch="tcn",
            Xg=np.zeros((48, 33, 2), dtype=np.float32),
            feat_cfg=SimpleNamespace(),
            model_cfg={"in_ch": 96},
            is_two_stream=False,
        )


def test_make_forward_fn_twostream_raises_on_feature_mismatch(monkeypatch):
    monkeypatch.setattr(
        "core.features.split_gcn_two_stream",
        lambda _xg, _feat: (
            np.zeros((48, 33, 5), dtype=np.float32),
            np.zeros((48, 33, 2), dtype=np.float32),
        ),
    )

    with __import__("pytest").raises(RuntimeError, match="joint feature mismatch"):
        dr._make_forward_fn(
            torch=object(),
            model=object(),
            device=SimpleNamespace(type="cpu"),
            arch="gcn",
            Xg=np.zeros((48, 33, 8), dtype=np.float32),
            feat_cfg=SimpleNamespace(),
            model_cfg={"in_feats_j": 6, "in_feats_m": 2},
            is_two_stream=True,
        )


def test_predict_spec_reuses_feature_cache_for_same_feat_cfg(monkeypatch):
    class _TensorLike:
        def __init__(self, v):
            self.v = float(v)

        def cpu(self):
            return self

        def view(self, *_shape):
            return self

        def __getitem__(self, _idx):
            return self

        def item(self):
            return self.v

    class _FakeTorch:
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        @staticmethod
        def inference_mode():
            return _FakeTorch._Ctx()

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
    monkeypatch.setattr(dr, "_torch", lambda: _FakeTorch())
    monkeypatch.setattr(
        dr,
        "_load_model_and_cfg",
        lambda _spec: {
            "model": object(),
            "device": SimpleNamespace(type="cpu"),
            "feat_cfg": SimpleNamespace(),
            "is_two_stream": False,
        },
    )

    calls = {"build": 0}

    def _build_canonical_input(**_kwargs):
        calls["build"] += 1
        return np.zeros((48, 33, 2), dtype=np.float32), None

    monkeypatch.setattr("core.features.build_canonical_input", _build_canonical_input)
    monkeypatch.setattr(dr, "_make_forward_fn", lambda **_kwargs: (lambda: _TensorLike(0.5)))

    cache = {}
    kwargs = dict(
        spec_key="muvim_tcn",
        joints_xy=np.zeros((48, 33, 2), dtype=np.float32),
        conf=np.ones((48, 33), dtype=np.float32),
        fps=25.0,
        target_T=48,
        op_code="OP-2",
        use_mc=False,
    )
    dr.predict_spec(feature_cache=cache, **kwargs)
    dr.predict_spec(feature_cache=cache, **kwargs)

    assert calls["build"] == 1
    assert len(cache) == 1


def test_predict_spec_feature_cache_distinguishes_different_windows(monkeypatch):
    class _TensorLike:
        def __init__(self, v):
            self.v = float(v)

        def cpu(self):
            return self

        def view(self, *_shape):
            return self

        def __getitem__(self, _idx):
            return self

        def item(self):
            return self.v

    class _FakeTorch:
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        @staticmethod
        def inference_mode():
            return _FakeTorch._Ctx()

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
    monkeypatch.setattr(dr, "_torch", lambda: _FakeTorch())
    monkeypatch.setattr(
        dr,
        "_load_model_and_cfg",
        lambda _spec: {
            "model": object(),
            "device": SimpleNamespace(type="cpu"),
            "feat_cfg": SimpleNamespace(),
            "is_two_stream": False,
        },
    )

    calls = {"build": 0}

    def _build_canonical_input(**_kwargs):
        calls["build"] += 1
        return np.zeros((48, 33, 2), dtype=np.float32), None

    monkeypatch.setattr("core.features.build_canonical_input", _build_canonical_input)
    monkeypatch.setattr(dr, "_make_forward_fn", lambda **_kwargs: (lambda: _TensorLike(0.5)))

    cache = {}
    kwargs = dict(
        spec_key="muvim_tcn",
        conf=np.ones((48, 33), dtype=np.float32),
        fps=25.0,
        target_T=48,
        op_code="OP-2",
        use_mc=False,
        feature_cache=cache,
    )
    dr.predict_spec(joints_xy=np.zeros((48, 33, 2), dtype=np.float32), **kwargs)
    dr.predict_spec(joints_xy=np.ones((48, 33, 2), dtype=np.float32), **kwargs)

    assert calls["build"] == 2
    assert len(cache) == 2


def test_predict_spec_sanitizes_inputs_before_feature_builder(monkeypatch):
    class _TensorLike:
        def __init__(self, v):
            self.v = float(v)

        def cpu(self):
            return self

        def view(self, *_shape):
            return self

        def __getitem__(self, _idx):
            return self

        def item(self):
            return self.v

    class _FakeTorch:
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        @staticmethod
        def inference_mode():
            return _FakeTorch._Ctx()

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
    monkeypatch.setattr(dr, "_torch", lambda: _FakeTorch())
    monkeypatch.setattr(
        dr,
        "_load_model_and_cfg",
        lambda _spec: {
            "model": object(),
            "device": SimpleNamespace(type="cpu"),
            "feat_cfg": SimpleNamespace(),
            "is_two_stream": False,
        },
    )

    seen = {"j": None, "c": None}

    def _build_canonical_input(**kwargs):
        seen["j"] = kwargs.get("joints_xy")
        seen["c"] = kwargs.get("conf")
        return np.zeros((48, 33, 2), dtype=np.float32), None

    monkeypatch.setattr("core.features.build_canonical_input", _build_canonical_input)
    monkeypatch.setattr(dr, "_make_forward_fn", lambda **_kwargs: (lambda: _TensorLike(0.5)))

    out = dr.predict_spec(
        spec_key="muvim_tcn",
        joints_xy=np.array([[[np.nan, -1.0]]] * 48, dtype=np.float32),
        conf=np.array([[1.5]] * 48, dtype=np.float32),
        fps=25.0,
        target_T=48,
        op_code="OP-2",
        use_mc=False,
    )

    assert out["p_det"] == 0.5
    assert isinstance(seen["j"], np.ndarray)
    assert isinstance(seen["c"], np.ndarray)
    assert bool(np.isfinite(seen["j"]).all())
    assert bool(np.isfinite(seen["c"]).all())
    assert bool(np.all((seen["j"] >= 0.0) & (seen["j"] <= 1.0)))
    assert bool(np.all((seen["c"] >= 0.0) & (seen["c"] <= 1.0)))


def test_predict_spec_rejects_target_t_mismatch(monkeypatch):
    class _FakeTorch:
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        @staticmethod
        def inference_mode():
            return _FakeTorch._Ctx()

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
    monkeypatch.setattr(dr, "_torch", lambda: _FakeTorch())
    monkeypatch.setattr(
        dr,
        "_load_model_and_cfg",
        lambda _spec: {
            "model": object(),
            "device": SimpleNamespace(type="cpu"),
            "feat_cfg": SimpleNamespace(),
            "is_two_stream": False,
        },
    )

    with pytest.raises(ValueError, match="time length mismatch"):
        dr.predict_spec(
            spec_key="muvim_tcn",
            joints_xy=np.zeros((47, 33, 2), dtype=np.float32),
            conf=np.ones((47, 33), dtype=np.float32),
            fps=25.0,
            target_T=48,
            op_code="OP-2",
            use_mc=False,
        )


def test_predict_spec_rejects_conf_shape_mismatch(monkeypatch):
    class _FakeTorch:
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        @staticmethod
        def inference_mode():
            return _FakeTorch._Ctx()

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
    monkeypatch.setattr(dr, "_torch", lambda: _FakeTorch())
    monkeypatch.setattr(
        dr,
        "_load_model_and_cfg",
        lambda _spec: {
            "model": object(),
            "device": SimpleNamespace(type="cpu"),
            "feat_cfg": SimpleNamespace(),
            "is_two_stream": False,
        },
    )

    with pytest.raises(ValueError, match="conf shape mismatch"):
        dr.predict_spec(
            spec_key="muvim_tcn",
            joints_xy=np.zeros((48, 33, 2), dtype=np.float32),
            conf=np.ones((48, 32), dtype=np.float32),
            fps=25.0,
            target_T=48,
            op_code="OP-2",
            use_mc=False,
        )


def test_predict_spec_assume_sanitized_inputs_skips_clip(monkeypatch):
    class _TensorLike:
        def __init__(self, value):
            self._v = float(value)

        def view(self, _n):
            return self

        def __getitem__(self, _idx):
            return self

        def item(self):
            return self._v

    class _FakeTorch:
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        @staticmethod
        def inference_mode():
            return _FakeTorch._Ctx()

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
    monkeypatch.setattr(dr, "_torch", lambda: _FakeTorch())
    monkeypatch.setattr(
        dr,
        "_load_model_and_cfg",
        lambda _spec: {
            "model": object(),
            "device": SimpleNamespace(type="cpu"),
            "feat_cfg": SimpleNamespace(),
            "is_two_stream": False,
        },
    )

    seen = {"j": None, "c": None}

    def _build_canonical_input(**kwargs):
        seen["j"] = kwargs.get("joints_xy")
        seen["c"] = kwargs.get("conf")
        return np.zeros((48, 33, 2), dtype=np.float32), None

    monkeypatch.setattr("core.features.build_canonical_input", _build_canonical_input)
    monkeypatch.setattr(dr, "_make_forward_fn", lambda **_kwargs: (lambda: _TensorLike(0.5)))

    joints = np.full((48, 33, 2), 1.5, dtype=np.float32)
    conf = np.full((48, 33), -0.5, dtype=np.float32)
    out = dr.predict_spec(
        spec_key="muvim_tcn",
        joints_xy=joints,
        conf=conf,
        fps=25.0,
        target_T=48,
        op_code="OP-2",
        use_mc=False,
        assume_sanitized_inputs=True,
    )

    assert out["p_det"] == 0.5
    assert isinstance(seen["j"], np.ndarray)
    assert isinstance(seen["c"], np.ndarray)
    assert float(seen["j"][0, 0, 0]) == 1.5
    assert float(seen["c"][0, 0]) == -0.5


def test_online_alert_tracker_transition_flow():
    trk = OnlineAlertTracker({"tau_low": 0.5, "tau_high": 0.8, "k": 2, "n": 3, "cooldown_s": 2.0, "ema_alpha": 0.0})
    r1 = trk.step(0.9, 0.0)
    r2 = trk.step(0.95, 0.1)
    # After two highs in window with k=2, should start event.
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
    # _list_tables with dict rows
    core_mod._TABLE_CACHE = None
    c1 = _FakeConn(responses=[[{"Tables_in_x": "events"}, {"Tables_in_x": "models"}]])
    tables = core_mod._list_tables(c1)
    assert "events" in tables and "models" in tables

    # _cols / _has_col
    core_mod._COL_CACHE.clear()
    c2 = _FakeConn(responses=[[{"Field": "ts"}, {"Field": "score"}]])
    cols = core_mod._cols(c2, "events")
    assert "ts" in cols
    assert core_mod._has_col(c2, "events", "score") is True

    # _one_resident_id / _resident_exists
    c3 = _FakeConn(responses=[{"id": 11}, {"ok": 1}])
    assert core_mod._one_resident_id(c3) == 11
    assert core_mod._resident_exists(c3, 11) is True

    # model/op resolvers
    c4 = _FakeConn(
        responses=[
            {"id": 5},          # resolve model id by code
            {"code": "GCN"},    # resolve model code by id
            {"id": 3, "model_id": 5},  # resolve op id
        ]
    )
    assert core_mod._resolve_model_id(c4, "GCN") == 5
    assert core_mod._resolve_model_code(c4, 5) == "GCN"
    assert core_mod._resolve_op_id(c4, 5, 3) == 3

    # event column pickers
    monkeypatch.setattr(core_mod, "_cols", lambda _conn, _table: {"created_at", "p_fall"})
    assert core_mod._event_time_col(c4) == "created_at"
    assert core_mod._event_prob_col(c4) == "p_fall"


def test_core_clip_privacy_and_caregiver_table_paths(monkeypatch):
    c = _FakeConn(
        responses=[
            {"store_event_clips": 1, "anonymize_skeleton_data": 0},  # read privacy flags
        ]
    )
    monkeypatch.setattr(core_mod, "_ensure_system_settings_schema", lambda _c: None)
    monkeypatch.setattr(core_mod, "_detect_variants", lambda _c: {"settings": "v2", "events": "v1", "ops": "v1"})
    monkeypatch.setattr(core_mod, "_table_exists", lambda _c, t: t in {"system_settings"})

    store, anon = core_mod._read_clip_privacy_flags(c, resident_id=1)
    assert store is True
    assert anon is False

    # create caregivers table path
    c2 = _FakeConn()
    monkeypatch.setattr(core_mod, "_table_exists", lambda _c, t: False if t == "caregivers" else True)
    core_mod._ensure_caregivers_table(c2)
    assert c2.commits >= 1


def test_core_system_settings_schema_and_variants(monkeypatch):
    # Simulate system_settings table with one missing column to force ALTER.
    c = _FakeConn(
        responses=[
            [{"Tables_in_x": "system_settings"}],  # SHOW TABLES
            [{"Field": "fall_threshold"}],         # SHOW COLUMNS system_settings
        ]
    )
    core_mod._TABLE_CACHE = None
    core_mod._COL_CACHE.clear()
    core_mod._ensure_system_settings_schema(c)
    assert c.commits >= 1

    # variants path
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
    # _resolve_model_id: code miss -> family hit
    c = _FakeConn(
        responses=[
            {},         # first SELECT id FROM models WHERE code=...
            {"id": 7},  # second SELECT by family
        ]
    )
    assert core_mod._resolve_model_id(c, "GCN") == 7

    # _resolve_op_id mismatch model should return None
    c2 = _FakeConn(responses=[{"id": 4, "model_id": 123}])
    assert core_mod._resolve_op_id(c2, model_id=999, op_id=4) is None

    # _event_prob_col with no known columns
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
