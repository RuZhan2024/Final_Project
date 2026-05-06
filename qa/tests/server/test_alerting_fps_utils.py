from __future__ import annotations

import numpy as np

from fall_detection.core.alerting import times_from_windows, window_span_seconds


def test_times_from_windows_supports_start_center_and_end_modes():
    starts = np.asarray([0, 25], dtype=np.int32)
    ends = np.asarray([24, 49], dtype=np.int32)

    assert np.allclose(times_from_windows(starts, ends, 25.0, mode="start"), [0.0, 1.0])
    assert np.allclose(times_from_windows(starts, ends, 25.0, mode="end"), [0.96, 1.96])
    assert np.allclose(times_from_windows(starts, ends, 25.0, mode="center"), [0.48, 1.48])


def test_times_from_windows_uses_default_rate_for_invalid_fps():
    assert np.allclose(times_from_windows([0], [29], 0.0, mode="end"), [29.0 / 30.0])


def test_window_span_seconds_uses_inclusive_frame_indices():
    assert window_span_seconds(0, 24, 25.0) == 1.0
    assert window_span_seconds(10, 39, 0.0) == 1.0
