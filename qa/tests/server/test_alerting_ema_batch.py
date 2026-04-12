from __future__ import annotations

import numpy as np

from core.alerting import _ema_precompute_by_groups, _ema_smooth_matrix, ema_smooth


def test_ema_smooth_matrix_matches_rowwise_scalar():
    rng = np.random.default_rng(123)
    x = rng.uniform(0.0, 1.0, size=(5, 32)).astype(np.float32)
    y_mat = _ema_smooth_matrix(x, 0.2)
    y_ref = np.stack([ema_smooth(row, 0.2) for row in x], axis=0)
    assert np.allclose(y_mat, y_ref, atol=1e-6)


def test_ema_precompute_by_groups_matches_direct_indexing():
    rng = np.random.default_rng(9)
    probs = rng.uniform(0.0, 1.0, size=(24,)).astype(np.float32)
    groups = [
        ("v1", np.asarray([0, 1, 2, 3, 4], dtype=np.int64)),
        ("v2", np.asarray([5, 6, 7, 8, 9], dtype=np.int64)),
        ("v3", np.asarray([10, 11, 12], dtype=np.int64)),
        ("v4", np.asarray([13, 14, 15], dtype=np.int64)),
    ]
    out = _ema_precompute_by_groups(probs, groups, 0.35)
    for k, idx in groups:
        assert np.allclose(out[k], ema_smooth(probs[idx], 0.35), atol=1e-6)
