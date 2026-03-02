"""Temporal pose resampling helpers for data pipeline compatibility."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def resample_pose_window(
    *,
    raw_t_ms: Sequence[Any],
    raw_xy: Sequence[Any],
    raw_conf: Sequence[Any] | None = None,
    target_fps: float = 25.0,
    target_T: int = 48,
    window_end_t_ms: float | None = None,
    prevalidated_raw: bool = False,
) -> tuple[list[list[list[float]]], list[list[float]], float, float, float]:
    """Resample one pose segment into a fixed-T window at target FPS.

    Returns:
        (xy_out, conf_out, start_t_ms, end_t_ms, fps_est)
    """

    _ = prevalidated_raw  # compatibility arg

    t = np.asarray(raw_t_ms, dtype=np.float64).reshape(-1)
    if t.size == 0:
        return [], [], 0.0, 0.0, float(target_fps)

    xy = np.asarray(raw_xy, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[0] != t.shape[0] or xy.shape[2] != 2:
        return [], [], 0.0, 0.0, float(target_fps)

    if raw_conf is None:
        conf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
    else:
        conf = np.asarray(raw_conf, dtype=np.float32)
        if conf.ndim != 2 or conf.shape[:2] != xy.shape[:2]:
            conf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)

    t = np.maximum.accumulate(t)
    dt = np.diff(t)
    dt = dt[dt > 0]
    fps_est = float(1000.0 / np.median(dt)) if dt.size else float(target_fps)

    fps = float(target_fps) if float(target_fps) > 0 else 25.0
    T = int(target_T) if int(target_T) > 1 else int(xy.shape[0])
    dt_ms = 1000.0 / fps

    end_t = float(window_end_t_ms) if window_end_t_ms is not None else float(t[-1])
    start_t = end_t - (T - 1) * dt_ms
    query_t = start_t + np.arange(T, dtype=np.float64) * dt_ms

    xy_out = np.empty((T, xy.shape[1], 2), dtype=np.float32)
    conf_out = np.empty((T, conf.shape[1]), dtype=np.float32)
    for j in range(xy.shape[1]):
        xy_out[:, j, 0] = np.interp(query_t, t, xy[:, j, 0]).astype(np.float32)
        xy_out[:, j, 1] = np.interp(query_t, t, xy[:, j, 1]).astype(np.float32)
        conf_out[:, j] = np.interp(query_t, t, conf[:, j]).astype(np.float32)

    return (
        xy_out.tolist(),
        conf_out.tolist(),
        float(start_t),
        float(end_t),
        float(fps_est),
    )

