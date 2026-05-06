from __future__ import annotations

import pytest

from fall_detection.evaluation.fit_ops import pick_ops_from_sweep_conservative


def test_collapsed_ops_reorder_honors_op2_tie_break() -> None:
    sweep = {
        "thr": [0.40, 0.50, 0.60],
        "tau_low": [0.32, 0.40, 0.48],
        "precision": [1.0, 1.0, 1.0],
        "recall": [1.0, 1.0, 1.0],
        "f1": [1.0, 1.0, 1.0],
        "fa24h": [0.0, 0.0, 0.0],
        "mean_delay_s": [1.0, 2.0, 3.0],
        "median_delay_s": [1.0, 2.0, 3.0],
        "n_gt_events": [5, 5, 5],
        "n_alert_events": [5, 5, 5],
        "n_true_alerts": [5, 5, 5],
        "n_false_alerts": [0, 0, 0],
    }

    ops_min, _ = pick_ops_from_sweep_conservative(
        sweep,
        op1_recall=0.95,
        op3_fa24h=1.0,
        tie_break="min_thr",
        min_tau_high=0.4,
    )
    ops_max, _ = pick_ops_from_sweep_conservative(
        sweep,
        op1_recall=0.95,
        op3_fa24h=1.0,
        tie_break="max_thr",
        min_tau_high=0.4,
    )

    assert ops_min["OP2"]["tau_high"] == pytest.approx(0.40)
    assert ops_max["OP2"]["tau_high"] == pytest.approx(0.60)
