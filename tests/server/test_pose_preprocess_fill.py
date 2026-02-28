import numpy as np

from pose.preprocess_pose_npz import linear_fill_small_gaps


def test_linear_fill_small_gaps_fast_path_when_fully_valid():
    rng = np.random.default_rng(1)
    xy = rng.normal(size=(12, 33, 2)).astype(np.float32)
    conf = np.full((12, 33), 0.9, dtype=np.float32)

    out_xy, out_conf, filled = linear_fill_small_gaps(
        xy,
        conf,
        conf_thr=0.2,
        max_gap=3,
        fill_conf="thr",
    )

    assert np.array_equal(filled, np.zeros_like(filled, dtype=bool))
    assert np.allclose(out_xy, xy, atol=0.0, rtol=0.0)
    assert np.allclose(out_conf, conf, atol=0.0, rtol=0.0)


def test_linear_fill_small_gaps_fills_short_internal_gap():
    xy = np.zeros((6, 1, 2), dtype=np.float32)
    conf = np.full((6, 1), 1.0, dtype=np.float32)
    xy[:, 0, 0] = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.float32)
    xy[:, 0, 1] = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.float32)

    # Mark frame 2 as missing (internal gap length 1).
    conf[2, 0] = 0.0
    xy[2, 0, :] = np.nan

    out_xy, out_conf, filled = linear_fill_small_gaps(
        xy,
        conf,
        conf_thr=0.5,
        max_gap=1,
        fill_conf="thr",
    )

    assert bool(filled[2, 0])
    assert np.allclose(out_xy[2, 0], np.asarray([2.0, 2.0], dtype=np.float32))
    assert float(out_conf[2, 0]) >= 0.5
