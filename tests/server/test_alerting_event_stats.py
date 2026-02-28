import numpy as np

from core.alerting import (
    AlertEvent,
    _count_true_false_alert_events,
    _event_match_delay_and_alert_stats_arrays,
    _event_match_delay_and_alert_stats_indices,
    _event_match_delay_and_alert_stats,
    _match_gt_delays,
)


def test_event_match_delay_and_alert_stats_matches_reference():
    gt = [(2.0, 4.0), (8.0, 9.5), (12.0, 13.0)]
    alerts = [
        AlertEvent(0, 0, 1.0, 1.5, 0.7),
        AlertEvent(1, 3, 2.5, 4.2, 0.8),
        AlertEvent(4, 5, 8.8, 9.1, 0.9),
        AlertEvent(6, 7, 15.0, 16.0, 0.6),
    ]
    slack = 0.2

    matched, delays = _match_gt_delays(gt, alerts, slack_s=slack)
    true_a, false_a = _count_true_false_alert_events(alerts, gt, slack_s=slack)

    m2, dsum2, nd2, t2, f2 = _event_match_delay_and_alert_stats(gt, alerts, slack_s=slack)

    assert m2 == matched
    assert nd2 == len(delays)
    assert np.isclose(dsum2, float(sum(delays)))
    assert t2 == true_a
    assert f2 == false_a


def test_event_match_delay_and_alert_stats_edge_cases():
    gt = [(1.0, 2.0)]
    alerts = [AlertEvent(0, 0, 3.0, 4.0, 0.5)]

    m, dsum, nd, t, f = _event_match_delay_and_alert_stats(gt, [], slack_s=0.0)
    assert (m, dsum, nd, t, f) == (0, 0.0, 0, 0, 0)

    m, dsum, nd, t, f = _event_match_delay_and_alert_stats([], alerts, slack_s=0.0)
    assert (m, dsum, nd, t, f) == (0, 0.0, 0, 0, 1)


def test_event_match_indices_matches_array_time_variant():
    t = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5], dtype=np.float32)
    gt_start = np.asarray([0.4, 2.8], dtype=np.float32)
    gt_end = np.asarray([1.6, 3.4], dtype=np.float32)
    starts = np.asarray([1, 5], dtype=np.int32)
    ends = np.asarray([3, 6], dtype=np.int32)

    out_idx = _event_match_delay_and_alert_stats_indices(
        gt_start,
        gt_end,
        t,
        starts,
        ends,
        slack_s=0.1,
    )
    out_arr = _event_match_delay_and_alert_stats_arrays(
        gt_start,
        gt_end,
        t[starts],
        t[ends],
        slack_s=0.1,
    )
    assert out_idx == out_arr


def test_event_match_indices_matches_array_time_variant_randomized():
    rng = np.random.default_rng(1234)
    for _ in range(20):
        n_alert = int(rng.integers(1, 16))
        n_gt = int(rng.integers(1, 12))

        starts = []
        ends = []
        cur = 0
        for _i in range(n_alert):
            s = cur + int(rng.integers(0, 3))
            e = s + int(rng.integers(1, 4))
            starts.append(s)
            ends.append(e)
            cur = e + int(rng.integers(0, 3))

        t = np.arange(max(ends) + 3, dtype=np.float32) * 0.25
        starts_i = np.asarray(starts, dtype=np.int32)
        ends_i = np.asarray(ends, dtype=np.int32)

        gt_s = np.sort(rng.uniform(float(t[0]), float(t[-1]), size=(n_gt,)).astype(np.float32))
        gt_e = np.minimum(gt_s + rng.uniform(0.1, 1.2, size=(n_gt,)).astype(np.float32), float(t[-1]))

        out_idx = _event_match_delay_and_alert_stats_indices(
            gt_s,
            gt_e,
            t,
            starts_i,
            ends_i,
            slack_s=0.1,
        )
        out_arr = _event_match_delay_and_alert_stats_arrays(
            gt_s,
            gt_e,
            t[starts_i],
            t[ends_i],
            slack_s=0.1,
        )
        assert out_idx[0] == out_arr[0]
        assert np.isclose(out_idx[1], out_arr[1], atol=1e-6)
        assert out_idx[2:] == out_arr[2:]


def test_event_match_indices_single_gt_fast_path_parity():
    t = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    gt_start = np.asarray([0.8], dtype=np.float32)
    gt_end = np.asarray([1.6], dtype=np.float32)
    starts = np.asarray([0, 2, 4], dtype=np.int32)
    ends = np.asarray([1, 3, 4], dtype=np.int32)

    out_idx = _event_match_delay_and_alert_stats_indices(
        gt_start, gt_end, t, starts, ends, slack_s=0.1
    )
    out_arr = _event_match_delay_and_alert_stats_arrays(
        gt_start, gt_end, t[starts], t[ends], slack_s=0.1
    )
    assert out_idx == out_arr


def test_event_match_indices_single_alert_fast_path_parity():
    t = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    gt_start = np.asarray([0.2, 1.8], dtype=np.float32)
    gt_end = np.asarray([0.7, 2.3], dtype=np.float32)
    starts = np.asarray([3], dtype=np.int32)
    ends = np.asarray([4], dtype=np.int32)

    out_idx = _event_match_delay_and_alert_stats_indices(
        gt_start, gt_end, t, starts, ends, slack_s=0.1
    )
    out_arr = _event_match_delay_and_alert_stats_arrays(
        gt_start, gt_end, t[starts], t[ends], slack_s=0.1
    )
    assert out_idx == out_arr
