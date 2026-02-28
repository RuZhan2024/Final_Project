import numpy as np

from pose.preprocess_pose_npz import smooth_weighted_moving_average


def _naive_smooth_weighted_moving_average(xy: np.ndarray, conf: np.ndarray, conf_thr: float, k: int) -> np.ndarray:
    if k <= 1:
        return xy
    if k % 2 == 0:
        k += 1

    T, J, _ = xy.shape
    out = np.full_like(xy, np.nan, dtype=np.float32)

    w = np.where((conf >= conf_thr) & np.isfinite(conf), conf, 0.0).astype(np.float32)
    half = k // 2
    xy_pad = np.pad(xy, ((half, half), (0, 0), (0, 0)), mode="edge")
    w_pad = np.pad(w, ((half, half), (0, 0)), mode="edge")

    for t in range(T):
        x_win = xy_pad[t : t + k]
        w_win = w_pad[t : t + k]

        nan_mask = ~np.isfinite(x_win[..., 0]) | ~np.isfinite(x_win[..., 1])
        w_eff = np.where(nan_mask, 0.0, w_win).astype(np.float32, copy=False)

        denom = w_eff.sum(axis=0)
        ok = denom > 1e-8
        if not ok.any():
            continue

        x_safe = np.where(nan_mask[..., None], 0.0, x_win)
        num = (x_safe * w_eff[..., None]).sum(axis=0)
        out[t, ok, :] = (num[ok, :] / denom[ok, None]).astype(np.float32, copy=False)

    return out


def test_smooth_weighted_moving_average_matches_naive():
    rng = np.random.default_rng(42)
    T, J = 31, 33
    xy = rng.normal(0.0, 1.0, size=(T, J, 2)).astype(np.float32)
    conf = rng.uniform(0.0, 1.0, size=(T, J)).astype(np.float32)

    # Inject missing joints / malformed values similar to real extraction noise.
    xy[3:6, 5, :] = np.nan
    xy[10, 2, 1] = np.nan
    conf[8:11, 12] = 0.0
    conf[15, 7] = np.nan

    got = smooth_weighted_moving_average(xy, conf, conf_thr=0.2, k=6)  # even k path
    exp = _naive_smooth_weighted_moving_average(xy, conf, conf_thr=0.2, k=6)

    # Compare finite values only and ensure NaN pattern is preserved.
    assert np.array_equal(np.isfinite(got), np.isfinite(exp))
    mask = np.isfinite(exp)
    assert np.allclose(got[mask], exp[mask], atol=1e-5, rtol=1e-5)
