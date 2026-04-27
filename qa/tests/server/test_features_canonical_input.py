import numpy as np
import pytest

from core.features import FeatCfg, build_canonical_input, channel_layout, split_gcn_two_stream


def test_build_canonical_input_all_valid_keeps_conf_channel():
    T, V = 6, 33
    rng = np.random.default_rng(0)
    joints = rng.uniform(0.0, 1.0, size=(T, V, 2)).astype(np.float32)
    conf = rng.uniform(0.2, 1.0, size=(T, V)).astype(np.float32)
    mask = np.ones((T, V), dtype=bool)

    cfg = FeatCfg(
        center="none",
        use_motion=False,
        use_bone=False,
        use_bone_length=False,
        use_conf_channel=True,
        use_precomputed_mask=True,
    )

    X, m = build_canonical_input(joints, None, conf, mask, fps=25.0, feat_cfg=cfg)
    lo = channel_layout(cfg)

    assert X.shape == (T, V, 3)
    assert np.array_equal(m, mask)
    assert np.allclose(X[..., slice(*lo["xy"])], joints)
    assert np.allclose(X[..., lo["conf"][0]], conf)


def test_build_canonical_input_masks_invalid_joints_across_channels():
    T, V = 8, 33
    joints = np.ones((T, V, 2), dtype=np.float32)
    conf = np.full((T, V), 0.7, dtype=np.float32)
    mask = np.ones((T, V), dtype=bool)
    mask[2:5, 10] = False

    cfg = FeatCfg(
        center="none",
        use_motion=True,
        use_bone=False,
        use_bone_length=False,
        use_conf_channel=True,
        use_precomputed_mask=True,
        motion_scale_by_fps=False,
    )

    X, m = build_canonical_input(joints, None, conf, mask, fps=25.0, feat_cfg=cfg)
    lo = channel_layout(cfg)

    assert np.array_equal(m, mask)
    xy = X[..., slice(*lo["xy"])]
    mot = X[..., slice(*lo["motion"])]
    conf_ch = X[..., lo["conf"][0]]

    assert np.allclose(xy[2:5, 10], 0.0)
    assert np.allclose(mot[2:5, 10], 0.0)
    assert np.allclose(conf_ch[2:5, 10], 0.0)


def test_split_gcn_two_stream_fast_path_without_motion():
    T, V = 5, 33
    rng = np.random.default_rng(1)
    joints = rng.uniform(0.0, 1.0, size=(T, V, 2)).astype(np.float32)
    conf = rng.uniform(0.2, 1.0, size=(T, V)).astype(np.float32)
    mask = np.ones((T, V), dtype=bool)

    cfg = FeatCfg(
        center="none",
        use_motion=False,
        use_bone=True,
        use_bone_length=True,
        use_conf_channel=True,
        use_precomputed_mask=True,
    )
    X, _ = build_canonical_input(joints, None, conf, mask, fps=25.0, feat_cfg=cfg)
    xj, xm = split_gcn_two_stream(X, cfg)

    assert np.shares_memory(xj, X)
    assert xj.shape == X.shape
    assert xm.shape == (T, V, 2)
    assert np.allclose(xm, 0.0)


def test_split_gcn_two_stream_full_layout_values():
    T, V = 4, 33
    rng = np.random.default_rng(2)
    joints = rng.uniform(0.0, 1.0, size=(T, V, 2)).astype(np.float32)
    conf = rng.uniform(0.2, 1.0, size=(T, V)).astype(np.float32)
    mask = np.ones((T, V), dtype=bool)

    cfg = FeatCfg(
        center="none",
        use_motion=True,
        use_bone=True,
        use_bone_length=True,
        use_conf_channel=True,
        use_precomputed_mask=True,
    )
    X, _ = build_canonical_input(joints, None, conf, mask, fps=25.0, feat_cfg=cfg)
    lo = channel_layout(cfg)
    xj, xm = split_gcn_two_stream(X, cfg)

    exp_xj = np.concatenate(
        [
            X[..., slice(*lo["xy"])],
            X[..., slice(*lo["bone"])],
            X[..., slice(*lo["bone_len"])],
            X[..., slice(*lo["conf"])],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    exp_xm = X[..., slice(*lo["motion"])]

    assert xj.shape == exp_xj.shape
    assert xm.shape == exp_xm.shape
    assert np.allclose(xj, exp_xj, atol=1e-6, rtol=1e-6)
    assert np.allclose(xm, exp_xm, atol=1e-6, rtol=1e-6)


def test_build_canonical_input_pelvis_center_fast_path_matches_expected():
    T, V = 5, 33
    rng = np.random.default_rng(3)
    joints = rng.uniform(0.0, 1.0, size=(T, V, 2)).astype(np.float32)
    conf = np.ones((T, V), dtype=np.float32)
    mask = np.ones((T, V), dtype=bool)

    cfg = FeatCfg(
        center="pelvis",
        use_motion=False,
        use_bone=False,
        use_bone_length=False,
        use_conf_channel=False,
        use_precomputed_mask=True,
    )

    X, m = build_canonical_input(joints, None, conf, mask, fps=25.0, feat_cfg=cfg)
    expected_center = 0.5 * (joints[:, 23:24, :] + joints[:, 24:25, :])
    expected_xy = joints - expected_center

    lo = channel_layout(cfg)
    assert np.array_equal(m, mask)
    assert np.allclose(X[..., slice(*lo["xy"])], expected_xy, atol=1e-6, rtol=1e-6)


def test_build_canonical_input_zero_length_window():
    joints = np.zeros((0, 33, 2), dtype=np.float32)
    conf = np.zeros((0, 33), dtype=np.float32)
    mask = np.zeros((0, 33), dtype=bool)
    cfg = FeatCfg(
        center="none",
        use_motion=True,
        use_bone=False,
        use_bone_length=False,
        use_conf_channel=True,
        use_precomputed_mask=True,
    )

    X, m = build_canonical_input(joints, None, conf, mask, fps=25.0, feat_cfg=cfg)
    assert X.shape == (0, 33, 5)  # xy(2)+motion(2)+conf(1)
    assert m.shape == (0, 33)


def test_build_canonical_input_rejects_invalid_conf_shape():
    joints = np.zeros((2, 3, 2), dtype=np.float32)
    conf = np.zeros((2, 2), dtype=np.float32)  # wrong shape/size
    mask = np.ones((2, 3), dtype=bool)
    cfg = FeatCfg(use_conf_channel=True, use_precomputed_mask=True)
    with pytest.raises(ValueError, match="conf must have shape"):
        build_canonical_input(joints, None, conf, mask, fps=25.0, feat_cfg=cfg)


def test_build_canonical_input_rejects_invalid_motion_shape():
    joints = np.zeros((2, 3, 2), dtype=np.float32)
    motion = np.zeros((2, 3), dtype=np.float32)  # wrong shape/size
    conf = np.ones((2, 3), dtype=np.float32)
    mask = np.ones((2, 3), dtype=bool)
    cfg = FeatCfg(use_motion=True, use_precomputed_mask=True)
    with pytest.raises(ValueError, match="motion_xy must match joints shape"):
        build_canonical_input(joints, motion, conf, mask, fps=25.0, feat_cfg=cfg)


def test_build_canonical_input_rejects_invalid_mask_shape():
    joints = np.zeros((2, 3, 2), dtype=np.float32)
    conf = np.ones((2, 3), dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)  # wrong shape/size
    cfg = FeatCfg(use_precomputed_mask=True)
    with pytest.raises(ValueError, match="mask must have shape"):
        build_canonical_input(joints, None, conf, mask, fps=25.0, feat_cfg=cfg)


def test_build_canonical_input_assume_finite_fast_path_matches_default():
    T, V = 6, 33
    rng = np.random.default_rng(7)
    joints = rng.uniform(0.0, 1.0, size=(T, V, 2)).astype(np.float32)
    conf = rng.uniform(0.05, 1.0, size=(T, V)).astype(np.float32)
    cfg = FeatCfg(
        center="pelvis",
        use_motion=True,
        use_bone=True,
        use_bone_length=True,
        use_conf_channel=True,
        use_precomputed_mask=False,
        conf_gate=0.2,
    )

    X_ref, m_ref = build_canonical_input(joints, None, conf, None, fps=25.0, feat_cfg=cfg)
    X_fast, m_fast = build_canonical_input(
        joints,
        None,
        conf,
        None,
        fps=25.0,
        feat_cfg=cfg,
        assume_finite_xy=True,
        assume_finite_conf=True,
    )

    assert np.array_equal(m_fast, m_ref)
    assert np.allclose(X_fast, X_ref, atol=1e-6, rtol=1e-6)
