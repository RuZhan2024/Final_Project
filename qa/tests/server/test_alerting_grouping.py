from __future__ import annotations

import numpy as np

from core.alerting import _build_video_groups


def test_build_video_groups_preserves_first_seen_video_order_and_sorts_windows():
    vids = np.asarray(["v2", "v1", "v2", "v3", "v1", "v3", "v2"], dtype=object)
    ws = np.asarray([30, 20, 10, 15, 5, 1, 25], dtype=np.int32)

    groups = _build_video_groups(vids, ws)
    keys = [k for (k, _idx) in groups]
    assert keys == ["v2", "v1", "v3"]

    by_key = {k: idx for (k, idx) in groups}
    assert ws[by_key["v2"]].tolist() == [10, 25, 30]
    assert ws[by_key["v1"]].tolist() == [5, 20]
    assert ws[by_key["v3"]].tolist() == [1, 15]


def test_build_video_groups_fast_path_for_already_grouped_sorted_input():
    vids = np.asarray(["v1", "v1", "v2", "v2", "v3"], dtype=object)
    ws = np.asarray([1, 5, 2, 8, 3], dtype=np.int32)

    groups = _build_video_groups(vids, ws)
    keys = [k for (k, _idx) in groups]
    assert keys == ["v1", "v2", "v3"]

    by_key = {k: idx for (k, idx) in groups}
    assert by_key["v1"].tolist() == [0, 1]
    assert by_key["v2"].tolist() == [2, 3]
    assert by_key["v3"].tolist() == [4]


def test_build_video_groups_falls_back_when_video_id_reappears_later():
    # v1 appears, then v2, then v1 again -> not fast-path compatible.
    vids = np.asarray(["v1", "v2", "v1", "v2"], dtype=object)
    ws = np.asarray([3, 2, 1, 4], dtype=np.int32)

    groups = _build_video_groups(vids, ws)
    keys = [k for (k, _idx) in groups]
    assert keys == ["v1", "v2"]

    by_key = {k: idx for (k, idx) in groups}
    # Fallback path should still sort each group's windows by w_start.
    assert ws[by_key["v1"]].tolist() == [1, 3]
    assert ws[by_key["v2"]].tolist() == [2, 4]
