from __future__ import annotations

import numpy as np

from core.alerting import _robust_time_step_s, _robust_video_fps


def test_robust_video_fps_constant_fast_path():
    fps = np.full((48,), 25.0, dtype=np.float32)
    out = _robust_video_fps(fps, 30.0)
    assert float(out) == 25.0


def test_robust_video_fps_mixed_and_invalid_values():
    fps = np.asarray([np.nan, 24.0, 26.0, np.nan, 25.0], dtype=np.float32)
    out = _robust_video_fps(fps, 30.0)
    assert np.isclose(float(out), 25.0)

    fps_bad = np.asarray([np.nan, np.nan], dtype=np.float32)
    out_bad = _robust_video_fps(fps_bad, 30.0)
    assert float(out_bad) == 30.0


def test_robust_time_step_s_constant_fast_path():
    t = np.asarray([0.0, 0.4, 0.8, 1.2], dtype=np.float32)
    out = _robust_time_step_s(t)
    assert out is not None
    assert np.isclose(float(out), 0.4)


def test_robust_time_step_s_nonuniform_uses_median():
    t = np.asarray([0.0, 0.4, 1.4, 1.8], dtype=np.float32)
    # diffs = [0.4, 1.0, 0.4] -> median = 0.4
    out = _robust_time_step_s(t)
    assert out is not None
    assert np.isclose(float(out), 0.4)
