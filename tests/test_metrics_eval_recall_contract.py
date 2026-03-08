from __future__ import annotations

import numpy as np

from fall_detection.core.alerting import AlertCfg
from fall_detection.evaluation import metrics_eval


def test_event_recall_uses_matched_gt_not_true_alerts(monkeypatch) -> None:
    """Recall must be bounded by 1 even when many alert events overlap one GT event."""
    n = 20
    probs = np.asarray([0.95 if i % 2 == 0 else 0.10 for i in range(n)], dtype=np.float32)
    y_true = np.ones(n, dtype=np.int32)  # single long GT event
    vids = np.asarray(["vid_a"] * n)
    ws = np.arange(n, dtype=np.int32) * 12
    we = ws + 47  # inclusive
    fps = np.full(n, 25.0, dtype=np.float32)

    alert_cfg = AlertCfg(
        ema_alpha=0.2,
        k=1,
        n=1,
        tau_high=0.8,
        tau_low=0.7,
        cooldown_s=0.0,
        confirm=False,
    )

    def _fake_event_metrics_from_windows(*args, **kwargs):
        # Simulate one GT event matched by multiple true alerts.
        return (
            {
                "n_gt_events": 1,
                "n_matched_gt": 1,
                "n_alert_events": 3,
                "n_true_alerts": 3,
                "n_false_alerts": 0,
                "recall": 1.0,
                "mean_delay_s": 0.1,
            },
            {},
        )

    monkeypatch.setattr(metrics_eval, "event_metrics_from_windows", _fake_event_metrics_from_windows)

    out = metrics_eval._aggregate_event_counts(
        probs,
        vids,
        ws,
        we,
        fps,
        y_true,
        fps_default=25.0,
        alert_cfg=alert_cfg,
        merge_gap_s=1.0,
        overlap_slack_s=0.0,
        time_mode="center",
    )

    assert int(out["n_gt_events"]) == 1
    assert int(out["n_true_alerts"]) == 3
    assert int(out["n_matched_gt"]) == 1
    assert float(out["recall"]) == 1.0
    assert 0.0 <= float(out["f1"]) <= 1.0
