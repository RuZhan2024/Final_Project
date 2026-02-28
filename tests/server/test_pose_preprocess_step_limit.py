import numpy as np

from pose.preprocess_pose_npz import limit_step_displacement


def test_limit_step_displacement_clamps_large_jump():
    xy = np.asarray(
        [
            [[0.0, 0.0]],
            [[10.0, 0.0]],
            [[10.0, 0.0]],
        ],
        dtype=np.float32,
    )

    out = limit_step_displacement(xy, max_step=1.0)

    assert out.shape == xy.shape
    # first jump should be clamped from 10 -> 1
    assert np.isclose(float(out[1, 0, 0]), 1.0)
    # second step is from 1 -> 10, clamped again by causal rule
    assert np.isclose(float(out[2, 0, 0]), 2.0)


def test_limit_step_displacement_preserves_nan_missing():
    xy = np.asarray(
        [
            [[0.0, 0.0]],
            [[np.nan, np.nan]],
            [[5.0, 5.0]],
        ],
        dtype=np.float32,
    )
    out = limit_step_displacement(xy, max_step=1.0)
    assert np.isnan(out[1, 0, 0])
    assert np.isnan(out[1, 0, 1])
