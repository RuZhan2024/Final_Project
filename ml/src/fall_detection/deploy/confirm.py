#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight confirm-score helpers without torch/runtime imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class WindowRaw:
    joints_xy: np.ndarray
    motion_xy: Optional[np.ndarray]
    conf: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    fps: float
    meta: Any


def compute_confirm_scores(
    raw: WindowRaw,
    *,
    conf_thr: float = 0.20,
    last_s: float = 0.7,
    min_valid_ratio: float = 0.25,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute (lying_score, motion_score) for a window."""
    joints = np.asarray(raw.joints_xy, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[0] <= 1:
        return None, None

    T, V, _ = joints.shape

    if raw.mask is not None:
        m = np.asarray(raw.mask, dtype=bool)
    elif raw.conf is not None:
        m = np.asarray(raw.conf, dtype=np.float32) >= float(conf_thr)
    else:
        m = np.ones((T, V), dtype=bool)

    valid_ratio = m.sum(axis=1).astype(np.float32) / float(max(1, V))

    fps = float(raw.fps) if float(raw.fps) > 0 else 30.0
    last_n = int(max(1, round(float(last_s) * fps)))
    last_n = min(last_n, T)

    if float(np.mean(valid_ratio[-last_n:])) < float(min_valid_ratio):
        return None, None

    def mean_xy_and_ok(idxs):
        idxs = [int(i) for i in idxs if 0 <= int(i) < V]
        if not idxs:
            return None, None
        vm = m[:, idxs].astype(np.float32)
        cnt = vm.sum(axis=1)
        sub = joints[:, idxs, :]
        s = (sub * vm[..., None]).sum(axis=1)
        out = np.zeros((T, 2), dtype=np.float32)
        ok = cnt > 0
        out[ok] = s[ok] / cnt[ok, None]
        return out, ok

    hip_mid, hip_ok = mean_xy_and_ok([23, 24])
    sh_mid, sh_ok = mean_xy_and_ok([11, 12])
    if hip_mid is None or sh_mid is None:
        return None, None

    torso_ok = hip_ok & sh_ok
    if int(torso_ok.sum()) < max(1, int(0.10 * T)):
        return None, None

    v = sh_mid - hip_mid
    n = np.linalg.norm(v, axis=1) + 1e-8
    vy = np.abs(v[:, 1]) / n
    lying_frame = np.clip(1.0 - vy, 0.0, 1.0).astype(np.float32)
    lying_frame[~torso_ok] = np.nan

    motion = None
    if raw.motion_xy is not None:
        mm = np.asarray(raw.motion_xy, dtype=np.float32)
        if mm.shape == joints.shape:
            motion = mm
    if motion is None:
        motion = np.zeros_like(joints, dtype=np.float32)
        motion[1:] = joints[1:] - joints[:-1]
        motion[0] = 0.0

    speed = np.linalg.norm(motion, axis=-1)
    speed = speed * m.astype(np.float32)
    cnt_all = m.sum(axis=1).astype(np.float32) + 1e-6
    speed_mean = speed.sum(axis=1) / cnt_all

    lying_score = float(np.nanmean(lying_frame[-last_n:]))
    if not np.isfinite(lying_score):
        return None, None

    peak = float(np.percentile(speed_mean, 90))
    tail = float(np.mean(speed_mean[-last_n:]))
    motion_score = float(np.clip(tail / (peak + 1e-6), 0.0, 1.0))
    return lying_score, motion_score
