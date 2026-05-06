from __future__ import annotations

import numpy as np

from fall_detection.core.alerting import AlertCfg, classify_states, ema_smooth


def test_ema_smooth_matches_manual_trace():
    x = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    y = ema_smooth(x, 0.5)
    assert np.allclose(y, np.asarray([0.0, 0.5, 0.25, 0.625], dtype=np.float32))


def test_ema_smooth_alpha_zero_returns_copy():
    x = np.asarray([0.2, 0.4, 0.6], dtype=np.float32)
    y = ema_smooth(x, 0.0)
    assert np.array_equal(y, x)
    assert y is not x


def test_classify_states_uses_smoothed_probs_and_alert_policy():
    cfg = AlertCfg(ema_alpha=0.5, k=1, n=1, tau_high=0.6, tau_low=0.4, cooldown_s=0.0)
    probs = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    times = np.arange(probs.size, dtype=np.float32)

    states = classify_states(probs, times, cfg)

    assert np.allclose(states["ps"], np.asarray([0.0, 0.5, 0.75, 0.375], dtype=np.float32))
    assert states["clear"].tolist() == [True, False, False, False]
    assert states["suspect"].tolist() == [False, True, False, False]
    assert states["alert"].tolist() == [False, False, True, True]
