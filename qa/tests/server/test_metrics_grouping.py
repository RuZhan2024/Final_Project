import numpy as np

from core.metrics import (
    _group_fp_events_for_video,
    _group_fp_events_for_video_from_scores,
)


def test_group_fp_events_from_scores_matches_reference_random_case():
    rng = np.random.default_rng(7)
    starts = np.sort(rng.integers(0, 200, size=80).astype(np.int64))
    ends = starts + rng.integers(2, 20, size=80).astype(np.int64)
    probs = rng.uniform(0.0, 1.0, size=80).astype(np.float32)
    stride = 6

    for thr in (0.1, 0.3, 0.5, 0.7, 0.9):
        active = probs >= thr
        ref = _group_fp_events_for_video(starts[active], ends[active], stride, assume_sorted=True)
        got = _group_fp_events_for_video_from_scores(probs, starts, ends, thr, stride)
        assert got == ref


def test_group_fp_events_from_scores_handles_empty_and_singleton():
    p = np.array([0.2, 0.1], dtype=np.float32)
    s = np.array([0, 10], dtype=np.int64)
    e = np.array([5, 12], dtype=np.int64)

    assert _group_fp_events_for_video_from_scores(p, s, e, thr=0.9, stride_frames=3) == 0
    assert _group_fp_events_for_video_from_scores(p, s, e, thr=0.15, stride_frames=3) == 1
