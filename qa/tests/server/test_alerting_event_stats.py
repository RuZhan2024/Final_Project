from __future__ import annotations

import math

import numpy as np

from fall_detection.core.alerting import AlertCfg, event_metrics_from_windows


def test_event_metrics_counts_true_and_false_alert_events():
    cfg = AlertCfg(ema_alpha=0.0, k=1, n=1, tau_high=0.8, tau_low=0.5, cooldown_s=0.0)
    probs = np.asarray([0.1, 0.95, 0.95, 0.2, 0.1, 0.95, 0.95, 0.2], dtype=np.float32)
    y_true = np.asarray([0, 1, 1, 0, 0, 0, 0, 0], dtype=np.int32)
    times = np.arange(probs.size, dtype=np.float32)

    metrics, detail = event_metrics_from_windows(
        probs,
        y_true,
        times,
        cfg,
        duration_s=8.0,
        merge_gap_s=1.0,
        overlap_slack_s=0.0,
    )

    assert metrics.n_gt_events == 1
    assert metrics.n_alert_events == 2
    assert metrics.n_matched_gt == 1
    assert metrics.n_true_alerts == 1
    assert metrics.n_false_alerts == 1
    assert metrics.event_recall == 1.0
    assert metrics.event_precision == 0.5
    assert metrics.event_f1 == 2.0 / 3.0
    assert metrics.false_alerts_per_hour == 450.0
    assert detail["gt_events"] == [{"start_s": 1.0, "end_s": 2.0}]
    assert [(ev["start_idx"], ev["end_idx"]) for ev in detail["alert_events"]] == [(1, 3), (5, 7)]


def test_event_metrics_empty_inputs_return_nan_rates_and_zero_counts():
    cfg = AlertCfg()
    metrics, detail = event_metrics_from_windows([], [], [], cfg)

    assert metrics.n_gt_events == 0
    assert metrics.n_alert_events == 0
    assert math.isnan(metrics.event_recall)
    assert math.isnan(metrics.false_alerts_per_day)
    assert detail == {"gt_events": [], "alert_events": []}
