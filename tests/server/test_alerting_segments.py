from __future__ import annotations

import numpy as np

from core.alerting import (
    _count_non_confirm_segments_from_pers,
    _detect_non_confirm_segments_from_pers,
)


def test_non_confirm_segments_respect_cooldown_and_retrigger():
    # Dense persistence with low-threshold drops at i=2 and i=6.
    # Cooldown=2s should allow retrigger at i=4 and i=8.
    t = np.arange(9, dtype=np.float32)
    ps = np.asarray([0.95, 0.95, 0.40, 0.95, 0.95, 0.95, 0.40, 0.95, 0.95], dtype=np.float32)
    pers = np.ones((9,), dtype=bool)

    starts, ends = _detect_non_confirm_segments_from_pers(
        ps,
        t,
        tau_low=0.5,
        cooldown_s=2.0,
        pers=pers,
    )
    assert starts.tolist() == [0, 4, 8]
    assert ends.tolist() == [2, 6, 8]

    n = _count_non_confirm_segments_from_pers(
        ps,
        t,
        tau_low=0.5,
        cooldown_s=2.0,
        pers=pers,
    )
    assert n == 3


def test_non_confirm_segments_pers_idx_override_matches_default():
    rng = np.random.default_rng(99)
    n = 120
    t = np.arange(n, dtype=np.float32) * 0.2
    ps = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float32)
    pers = rng.uniform(0.0, 1.0, size=(n,)) > 0.6
    pers_idx = np.flatnonzero(pers)
    low_idx = np.flatnonzero(ps < 0.45)

    s0, e0 = _detect_non_confirm_segments_from_pers(
        ps, t, tau_low=0.45, cooldown_s=1.0, pers=pers, low_idx=low_idx
    )
    s1, e1 = _detect_non_confirm_segments_from_pers(
        ps, t, tau_low=0.45, cooldown_s=1.0, pers=pers, low_idx=low_idx, pers_idx=pers_idx
    )
    assert s0.tolist() == s1.tolist()
    assert e0.tolist() == e1.tolist()

    c0 = _count_non_confirm_segments_from_pers(
        ps, t, tau_low=0.45, cooldown_s=1.0, pers=pers, low_idx=low_idx
    )
    c1 = _count_non_confirm_segments_from_pers(
        ps, t, tau_low=0.45, cooldown_s=1.0, pers=pers, low_idx=low_idx, pers_idx=pers_idx
    )
    assert c0 == c1
