import numpy as np

from pose.preprocess_pose_npz import clip_xy_finite


def test_clip_xy_finite_clips_values_and_keeps_nan():
    xy = np.asarray(
        [
            [[-0.5, 0.2], [1.6, np.nan]],
            [[0.3, 2.1], [np.nan, -2.0]],
        ],
        dtype=np.float32,
    )

    out = clip_xy_finite(xy, 0.0, 1.0)

    assert out.shape == xy.shape
    assert np.isclose(float(out[0, 0, 0]), 0.0)
    assert np.isclose(float(out[0, 0, 1]), 0.2)
    assert np.isclose(float(out[0, 1, 0]), 1.0)
    assert np.isnan(out[0, 1, 1])
    assert np.isclose(float(out[1, 0, 0]), 0.3)
    assert np.isclose(float(out[1, 0, 1]), 1.0)
    assert np.isnan(out[1, 1, 0])
    assert np.isclose(float(out[1, 1, 1]), 0.0)
