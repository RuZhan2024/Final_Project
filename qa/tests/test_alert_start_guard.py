from __future__ import annotations

import numpy as np

from fall_detection.core.alerting import AlertCfg, detect_alert_events


def _base_cfg(**kwargs) -> AlertCfg:
    return AlertCfg(
        ema_alpha=0.0,
        k=1,
        n=1,
        tau_high=0.8,
        tau_low=0.5,
        cooldown_s=0.0,
        confirm=False,
        **kwargs,
    )


def test_start_guard_prefixes_parse_stringified_list() -> None:
    cfg = AlertCfg.from_dict(
        {
            "start_guard_max_lying": 0.2,
            "start_guard_prefixes": "['Coffee_room_02__Videos__']",
        }
    )
    assert cfg.start_guard_max_lying == 0.2
    assert cfg.start_guard_prefixes == ["Coffee_room_02__Videos__"]


def test_start_guard_applies_only_for_matching_prefix() -> None:
    probs = np.asarray([0.9, 0.9, 0.1], dtype=np.float32)
    times = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    lying = np.asarray([0.95, 0.95, 0.95], dtype=np.float32)

    cfg = _base_cfg(start_guard_max_lying=0.2, start_guard_prefixes=["Coffee_room_02__Videos__"])

    _active_hit, events_hit = detect_alert_events(
        probs,
        times,
        cfg,
        lying_score=lying,
        video_id="Coffee_room_02__Videos__video__52_",
    )
    assert len(events_hit) == 0

    _active_miss, events_miss = detect_alert_events(
        probs,
        times,
        cfg,
        lying_score=lying,
        video_id="Home_01__Videos__video__4_",
    )
    assert len(events_miss) == 1


def test_start_guard_not_applied_without_video_id_when_prefixes_set() -> None:
    probs = np.asarray([0.9, 0.9, 0.1], dtype=np.float32)
    times = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    lying = np.asarray([0.95, 0.95, 0.95], dtype=np.float32)

    cfg = _base_cfg(start_guard_max_lying=0.2, start_guard_prefixes=["Coffee_room_02__Videos__"])
    _active, events = detect_alert_events(probs, times, cfg, lying_score=lying, video_id=None)
    assert len(events) == 1
