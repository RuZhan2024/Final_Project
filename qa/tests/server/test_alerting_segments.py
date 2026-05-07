from __future__ import annotations

import numpy as np

from fall_detection.core.alerting import AlertCfg, detect_alert_events


def test_detect_alert_events_respects_cooldown_and_retrigger():
    cfg = AlertCfg(ema_alpha=0.0, k=1, n=1, tau_high=0.9, tau_low=0.5, cooldown_s=2.0)
    times = np.arange(9, dtype=np.float32)
    probs = np.asarray([0.95, 0.95, 0.40, 0.95, 0.95, 0.95, 0.40, 0.95, 0.95], dtype=np.float32)

    active, events = detect_alert_events(probs, times, cfg)

    assert active.tolist() == [True, True, True, False, True, True, True, False, True]
    assert [(ev.start_idx, ev.end_idx) for ev in events] == [(0, 2), (4, 6), (8, 8)]


def test_detect_alert_events_requires_matching_probability_and_time_lengths():
    cfg = AlertCfg()

    try:
        detect_alert_events([0.1, 0.2], [0.0], cfg)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched probability/time lengths")
