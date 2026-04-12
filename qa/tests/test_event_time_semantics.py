from __future__ import annotations

import numpy as np

from fall_detection.core.alerting import times_from_windows, window_span_seconds
from fall_detection.core.features import read_window_npz


def test_times_from_windows_center_matches_inclusive_midpoint() -> None:
    ws = np.asarray([0, 12, 24], dtype=np.int32)
    we = np.asarray([47, 59, 71], dtype=np.int32)  # inclusive
    fps = 25.0
    t = times_from_windows(ws, we, fps, mode="center")
    expected = (ws + we) * 0.5 / fps
    np.testing.assert_allclose(t, expected, rtol=0.0, atol=1e-7)


def test_window_span_seconds_uses_inclusive_w_end() -> None:
    # 0..47 is 48 frames; at 25 FPS this is 1.92s
    assert np.isclose(window_span_seconds(0, 47, 25.0), 48.0 / 25.0)
    # 10..21 is 12 frames; at 24 FPS this is 0.5s
    assert np.isclose(window_span_seconds(10, 21, 24.0), 12.0 / 24.0)


def test_read_window_npz_meta_duration_consistency(tmp_path) -> None:
    fp = tmp_path / "w.npz"
    T, V = 48, 17
    np.savez_compressed(
        fp,
        joints=np.zeros((T, V, 2), dtype=np.float32),
        motion=np.zeros((T, V, 2), dtype=np.float32),
        conf=np.ones((T, V), dtype=np.float32),
        mask=np.ones((T, V), dtype=np.uint8),
        y=np.int32(0),
        fps=np.float32(25.0),
        video_id=np.array("vid"),
        w_start=np.int32(0),
        w_end=np.int32(T - 1),
    )
    _j, _m, _c, _mk, fps, meta = read_window_npz(str(fp), fps_default=25.0)
    dur_s = window_span_seconds(meta.w_start, meta.w_end, fps)
    assert np.isclose(dur_s, T / 25.0)

