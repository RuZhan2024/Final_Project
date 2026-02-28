import numpy as np

from pose.preprocess_pose_npz import compute_masks, normalize_body_centric


def test_compute_masks_zero_joint_dimension_no_nan():
    xy = np.zeros((3, 0, 2), dtype=np.float32)
    conf = np.zeros((3, 0), dtype=np.float32)
    joint_mask, frame_mask, valid_ratio = compute_masks(xy, conf, conf_thr=0.2)

    assert joint_mask.shape == (3, 0)
    assert frame_mask.shape == (3,)
    assert valid_ratio.shape == (3,)
    assert bool(np.isfinite(valid_ratio).all())
    assert np.all(valid_ratio == 0.0)


def test_normalize_body_centric_handles_small_joint_count():
    # Only 2 joints -> missing MP shoulder/hip indices used by torso normalization.
    xy = np.array(
        [
            [[0.1, 0.2], [0.2, 0.3]],
            [[0.2, 0.1], [0.3, 0.4]],
        ],
        dtype=np.float32,
    )
    conf = np.ones((2, 2), dtype=np.float32)

    out, meta = normalize_body_centric(
        xy,
        conf,
        conf_thr=0.2,
        mode="torso",
        pelvis_fill="nearest",
        rotate="shoulders",
    )

    assert out.shape == xy.shape
    assert out.dtype == np.float32
    assert isinstance(meta, dict)
    assert bool(np.isfinite(out).all())
