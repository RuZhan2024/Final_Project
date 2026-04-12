import math

import numpy as np

from core.alerting import AlertCfg, detect_alert_events_from_smoothed, sweep_alert_policy_from_windows
from core.metrics import sweep_with_fa24h
from core.confirm import lying_score_window, motion_score_window, confirm_scores_window


def test_detect_alert_events_from_smoothed_ignores_confirm_scores_when_disabled():
    cfg = AlertCfg(k=1, n=1, tau_high=0.9, tau_low=0.6, confirm=False)
    ps = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ts = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    # Deliberately mismatched lengths should not matter when confirm=False.
    active, events = detect_alert_events_from_smoothed(
        ps,
        ts,
        cfg,
        lying_score=[0.9],
        motion_score=[0.1, 0.2],
    )

    assert active.dtype == bool
    assert active.shape == ps.shape
    assert not active.any()
    assert events == []


def test_detect_alert_events_from_smoothed_length_mismatch_raises():
    cfg = AlertCfg()
    ps = np.array([0.1, 0.2], dtype=np.float32)
    ts = np.array([0.0], dtype=np.float32)
    try:
        detect_alert_events_from_smoothed(ps, ts, cfg)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected ValueError for length mismatch")


def test_sweep_with_fa24h_fast_paths_match_expected_shape_and_rates():
    probs = np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float32)
    y_true = np.array([0, 0, 0, 0], dtype=np.int32)
    vids = np.array(["v1", "v1", "v1", "v1"])
    ws = np.array([0, 2, 20, 22], dtype=np.int64)
    we = np.array([1, 3, 21, 23], dtype=np.int64)
    fps = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)

    sweep, meta = sweep_with_fa24h(
        probs,
        y_true,
        vids,
        ws,
        we,
        fps,
        thr_min=0.1,
        thr_max=0.9,
        thr_step=0.4,
        stride_frames_hint=2,
    )

    assert meta["stride_frames_used"] == 2
    assert len(sweep["thr"]) == 3
    assert len(sweep["fa24h"]) == 3

    # thr=0.1: all windows active -> two grouped FP events in this stream.
    assert sweep["fa24h"][0] > 0.0
    # thr>=0.5: no window active -> no FP events/day.
    assert sweep["fa24h"][1] == 0.0
    assert sweep["fa24h"][2] == 0.0


def test_sweep_alert_policy_from_windows_executes_and_returns_series():
    probs = np.array([0.1, 0.8, 0.9, 0.2], dtype=np.float32)
    y_true = np.array([0, 1, 1, 0], dtype=np.int32)
    vids = np.array(["v1", "v1", "v1", "v1"])
    ws = np.array([0, 12, 24, 36], dtype=np.int32)
    we = np.array([11, 23, 35, 47], dtype=np.int32)
    fps = np.array([25.0, 25.0, 25.0, 25.0], dtype=np.float32)
    cfg = AlertCfg(ema_alpha=0.2, k=1, n=1, tau_high=0.8, tau_low=0.6, confirm=False)

    sweep, meta = sweep_alert_policy_from_windows(
        probs,
        y_true,
        vids,
        ws,
        we,
        fps,
        alert_base=cfg,
        thr_min=0.5,
        thr_max=0.9,
        thr_step=0.2,
    )

    assert meta["n_videos"] == 1
    assert len(sweep["thr"]) == 3
    assert len(sweep["recall"]) == 3
    assert len(sweep["fa24h"]) == 3


def test_confirm_scores_window_matches_individual_scores():
    T, V = 12, 33
    joints = np.zeros((T, V, 2), dtype=np.float32)
    mask = np.ones((T, V), dtype=np.uint8)

    # Construct simple synthetic motion and pose geometry.
    for t in range(T):
        joints[t, 11] = [0.2 + 0.001 * t, 0.4]
        joints[t, 12] = [0.4 + 0.001 * t, 0.4]
        joints[t, 23] = [0.2 + 0.001 * t, 0.6]
        joints[t, 24] = [0.4 + 0.001 * t, 0.6]
        joints[t, 27] = [0.2 + 0.001 * t, 0.9]
        joints[t, 28] = [0.4 + 0.001 * t, 0.9]

    ls0 = lying_score_window(joints, mask, 25.0, tail_s=0.5, smooth="median")
    ms0 = motion_score_window(joints, mask, 25.0, tail_s=0.5, smooth="median")
    ls1, ms1 = confirm_scores_window(joints, mask, 25.0, tail_s=0.5, smooth="median")

    assert math.isclose(ls0, ls1, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(ms0, ms1, rel_tol=1e-6, abs_tol=1e-6)


def test_confirm_scores_window_all_missing_fast_path():
    T, V = 8, 33
    joints = np.full((T, V, 2), np.nan, dtype=np.float32)
    mask = np.zeros((T, V), dtype=np.uint8)
    ls, ms = confirm_scores_window(joints, mask, 25.0, tail_s=0.5, smooth="median")
    assert ls == 0.0
    assert math.isinf(ms)


def test_sweep_confirm_without_scores_matches_non_confirm():
    probs = np.array([0.1, 0.7, 0.9, 0.85, 0.2, 0.1], dtype=np.float32)
    y_true = np.array([0, 1, 1, 1, 0, 0], dtype=np.int32)
    vids = np.array(["v1"] * 6)
    ws = np.array([0, 10, 20, 30, 40, 50], dtype=np.int32)
    we = np.array([9, 19, 29, 39, 49, 59], dtype=np.int32)
    fps = np.array([25.0] * 6, dtype=np.float32)

    cfg_no = AlertCfg(ema_alpha=0.2, k=1, n=1, tau_high=0.8, tau_low=0.6, confirm=False)
    cfg_yes = AlertCfg(ema_alpha=0.2, k=1, n=1, tau_high=0.8, tau_low=0.6, confirm=True)

    sweep_no, _meta_no = sweep_alert_policy_from_windows(
        probs,
        y_true,
        vids,
        ws,
        we,
        fps,
        alert_base=cfg_no,
        thr_min=0.5,
        thr_max=0.9,
        thr_step=0.2,
    )
    sweep_yes, _meta_yes = sweep_alert_policy_from_windows(
        probs,
        y_true,
        vids,
        ws,
        we,
        fps,
        alert_base=cfg_yes,
        thr_min=0.5,
        thr_max=0.9,
        thr_step=0.2,
        lying_score=None,
        motion_score=None,
    )

    for k in ("recall", "fa24h", "f1", "n_alert_events", "n_false_alerts"):
        a = np.asarray(sweep_no[k], dtype=np.float64)
        b = np.asarray(sweep_yes[k], dtype=np.float64)
        assert np.allclose(a, b, atol=1e-9, rtol=1e-9, equal_nan=True)
