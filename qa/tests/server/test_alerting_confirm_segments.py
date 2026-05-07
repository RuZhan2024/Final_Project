from __future__ import annotations

import numpy as np

from fall_detection.core.alerting import AlertCfg, detect_alert_events


def test_confirm_stage_promotes_pending_alert_when_scores_pass_gate():
    cfg = AlertCfg(
        ema_alpha=0.0,
        k=1,
        n=1,
        tau_high=0.8,
        tau_low=0.5,
        cooldown_s=0.0,
        confirm=True,
        confirm_s=2.0,
        confirm_require_low=False,
        confirm_min_lying=0.7,
        confirm_max_motion=0.2,
    )
    probs = np.asarray([0.1, 0.95, 0.95, 0.95, 0.2], dtype=np.float32)
    times = np.arange(probs.size, dtype=np.float32)
    lying = np.asarray([np.nan, 0.2, 0.8, 0.8, 0.8], dtype=np.float32)
    motion = np.asarray([np.nan, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)

    active, events = detect_alert_events(probs, times, cfg, lying_score=lying, motion_score=motion)

    assert active.tolist() == [False, False, True, True, True]
    assert [(ev.start_idx, ev.end_idx) for ev in events] == [(1, 4)]
    assert events[0].start_time_s == 1.0
    assert events[0].end_time_s == 4.0


def test_confirm_stage_drops_pending_alert_when_gate_never_passes():
    cfg = AlertCfg(
        ema_alpha=0.0,
        k=1,
        n=1,
        tau_high=0.8,
        tau_low=0.5,
        cooldown_s=0.0,
        confirm=True,
        confirm_s=1.0,
        confirm_require_low=False,
        confirm_min_lying=0.7,
        confirm_max_motion=0.2,
    )
    probs = np.asarray([0.1, 0.95, 0.95, 0.95, 0.95], dtype=np.float32)
    times = np.arange(probs.size, dtype=np.float32)
    lying = np.full_like(probs, 0.2)
    motion = np.full_like(probs, 0.1)

    active, events = detect_alert_events(probs, times, cfg, lying_score=lying, motion_score=motion)

    assert not active.any()
    assert events == []


def test_confirm_without_extra_scores_uses_probability_policy():
    probs = np.asarray([0.1, 0.95, 0.95, 0.2], dtype=np.float32)
    times = np.arange(probs.size, dtype=np.float32)
    base = dict(ema_alpha=0.0, k=1, n=1, tau_high=0.8, tau_low=0.5, cooldown_s=0.0)

    active_no, events_no = detect_alert_events(probs, times, AlertCfg(**base, confirm=False))
    active_yes, events_yes = detect_alert_events(probs, times, AlertCfg(**base, confirm=True))

    assert active_yes.tolist() == active_no.tolist()
    assert [ev.to_dict() for ev in events_yes] == [ev.to_dict() for ev in events_no]
