#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/confirm.py

Confirm-stage heuristic signals (lying + motion).

These are intentionally simple and robust:
  - lying_score in [0,1] where 1 means "very horizontal/on ground".
  - motion_score >= 0 where smaller means "more still".

They are designed to be computed on the fly from the same joints used for
classification. No ML model is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ConfirmCfg:
    """Tail-window heuristic settings shared by lying and motion confirmation.

    These controls intentionally stay small because confirmation is a secondary
    gate on top of model scores, not a learned subsystem with its own training
    lifecycle.
    """
    tail_s: float = 1.0  # use last ~1s
    smooth: str = "median"  # median|mean


def _safe_fps(fps: float, *, default: float = 30.0) -> float:
    """Normalize runtime/training FPS inputs to a positive finite value."""
    try:
        f = float(fps)
    except Exception:
        f = default
    if not np.isfinite(f) or f <= 0:
        f = default
    return f


def _tail_n(T: int, fps: float, tail_s: float) -> int:
    """Convert tail duration into a bounded frame count for one window."""
    n = int(np.ceil(_safe_fps(fps) * float(tail_s)))
    n = max(1, min(int(T), n))
    return n


def _smooth_1d(x: np.ndarray, how: str) -> float:
    """Aggregate a 1D tail signal with the configured robust smoother."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    how = str(how).lower()
    if how == "mean":
        return float(np.mean(x))
    return float(np.median(x))


def _tail_finite_median(x: np.ndarray, sl: slice, *, default: float) -> float:
    """Median over finite tail values with an explicit fallback."""
    xx = np.asarray(x, dtype=np.float32).reshape(-1)
    tail = xx[sl]
    tail = tail[np.isfinite(tail)]
    if tail.size == 0:
        return float(default)
    return float(np.median(tail))


def _rowwise_nanmedian_or_default(x: np.ndarray, *, default: float) -> np.ndarray:
    """Row-wise nanmedian without warnings; all-NaN rows fall back to `default`."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("expected 2D array for row-wise nanmedian")
    out = np.full((arr.shape[0],), float(default), dtype=np.float32)
    finite = np.isfinite(arr)
    valid = finite.any(axis=1)
    if valid.any():
        out[valid] = np.nanmedian(arr[valid], axis=1).astype(np.float32)
    return out


def _finite_or_default(x: float, *, default: float) -> float:
    """Return finite scalar x, otherwise default."""
    xv = float(x)
    if not np.isfinite(xv):
        return float(default)
    return xv


def _bbox_hw(j: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame bbox height/width from visible joints.

    Safe against all-NaN frames (no warnings); returns NaNs for those frames.
    """
    # j: [T,V,2], m: [T,V]
    T, V, _ = j.shape
    m = m.astype(bool)
    m = m & np.isfinite(j).all(axis=2)
    valid = m.any(axis=1)  # [T]

    # set invalid joints to NaN so min/max ignore them
    jx = np.where(m[..., None], j, np.nan)

    xmin = np.full((T,), np.nan, dtype=np.float32)
    xmax = np.full((T,), np.nan, dtype=np.float32)
    ymin = np.full((T,), np.nan, dtype=np.float32)
    ymax = np.full((T,), np.nan, dtype=np.float32)

    if valid.any():
        vv = valid
        xmin[vv] = np.nanmin(jx[vv, :, 0], axis=1).astype(np.float32)
        xmax[vv] = np.nanmax(jx[vv, :, 0], axis=1).astype(np.float32)
        ymin[vv] = np.nanmin(jx[vv, :, 1], axis=1).astype(np.float32)
        ymax[vv] = np.nanmax(jx[vv, :, 1], axis=1).astype(np.float32)

    h = (ymax - ymin).astype(np.float32)
    w = (xmax - xmin).astype(np.float32)
    return h, w


def _joint_idx(V: int, idx: int) -> Optional[int]:
    """Return a valid joint index or None for skeletons with fewer joints."""
    return int(idx) if 0 <= int(idx) < int(V) else None


def _mean_of(j: np.ndarray, m: np.ndarray, ids: Iterable[int]) -> np.ndarray:
    """Compute per-frame mean of given joints; returns NaNs if none valid.

    Avoids RuntimeWarning for frames where all joints are missing.
    """
    T, V, _ = j.shape
    pts = []
    for k in ids:
        kk = _joint_idx(V, k)
        if kk is None:
            continue
        pts.append(j[:, kk, :])
    if not pts:
        return np.full((T, 2), np.nan, dtype=np.float32)

    P = np.stack(pts, axis=1)  # [T,K,2]
    Mm = np.stack(
        [m[:, _joint_idx(V, k)] if _joint_idx(V, k) is not None else np.zeros((T,), bool) for k in ids],
        axis=1,
    ).astype(bool)
    Mm = Mm & np.isfinite(P).all(axis=2)

    out = np.full((T, 2), np.nan, dtype=np.float32)
    cnt = Mm.sum(axis=1).astype(np.float32)  # [T]
    valid = cnt > 0
    if valid.any():
        summed = np.where(Mm[..., None], P, 0.0).sum(axis=1)  # [T,2]
        out[valid] = (summed[valid] / cnt[valid, None]).astype(np.float32)
    return out


def lying_score_window(
    joints_xy: np.ndarray,
    mask: np.ndarray,
    fps: float,
    *,
    tail_s: float = 1.0,
    smooth: str = "median",
) -> float:
    """Window-level lying score in [0,1].

    The score combines torso orientation and person-bbox aspect ratio over the
    tail of the window. Higher values mean the subject looks more horizontal.
    """
    j = np.asarray(joints_xy, dtype=np.float32)
    m = np.asarray(mask).astype(bool)
    if j.ndim != 3 or j.shape[-1] != 2:
        raise ValueError("joints_xy must be [T,V,2]")

    T, V, _ = j.shape
    if not m.any():
        # No observable joints in this window: signal is unavailable.
        return float("nan")
    n_tail = _tail_n(T, fps, tail_s)
    sl = slice(T - n_tail, T)

    # Support both MediaPipe-33 and COCO-like layouts because evaluation data
    # and runtime exports do not always share the same joint count.
    if V >= 33:
        sh = _mean_of(j, m, [11, 12])
        hip = _mean_of(j, m, [23, 24])
    else:
        sh = _mean_of(j, m, [5, 6])
        hip = _mean_of(j, m, [11, 12])

    torso = sh - hip  # [T,2]
    dx = np.abs(torso[:, 0])
    dy = np.abs(torso[:, 1])
    norm = np.sqrt(dx * dx + dy * dy) + 1e-6
    # horizontalness: 1 when dy ~= 0, 0 when dx ~= 0
    torso_h = 1.0 - np.clip(dy / norm, 0.0, 1.0)

    h, w = _bbox_hw(j, m)
    ratio = h / (w + 1e-6)
    # map bbox ratio to [0,1]: <=0.6 => 1, >=1.5 => 0
    bbox_h = np.clip((1.5 - ratio) / (1.5 - 0.6), 0.0, 1.0)

    score = 0.6 * torso_h + 0.4 * bbox_h
    s = _smooth_1d(score[sl], smooth)
    if not np.isfinite(s):
        return float("nan")
    return float(np.clip(s, 0.0, 1.0))


def motion_score_window(
    joints_xy: np.ndarray,
    mask: np.ndarray,
    fps: float,
    *,
    tail_s: float = 1.0,
    smooth: str = "median",
) -> float:
    """Window-level motion score (smaller = more still).

    Returns a dimensionless speed normalized by torso length.
    """
    j = np.asarray(joints_xy, dtype=np.float32)
    m = np.asarray(mask).astype(bool)
    if j.ndim != 3 or j.shape[-1] != 2:
        raise ValueError("joints_xy must be [T,V,2]")

    T, V, _ = j.shape
    fps = _safe_fps(fps)
    n_tail = _tail_n(T, fps, tail_s)
    sl = slice(T - n_tail, T)

    # Use torso/leg anchors that stay relatively stable under pose jitter;
    # wrist-dominant motion is too noisy for confirm-stage gating.
    if V >= 33:
        ids = [11, 12, 23, 24, 27, 28]
        sh = _mean_of(j, m, [11, 12])
        hip = _mean_of(j, m, [23, 24])
    else:
        ids = [5, 6, 11, 12, 15, 16]
        sh = _mean_of(j, m, [5, 6])
        hip = _mean_of(j, m, [11, 12])

    torso_len = np.sqrt(np.sum((sh - hip) ** 2, axis=1)) + 1e-6
    torso_len = _tail_finite_median(torso_len, sl, default=1.0)
    if not np.isfinite(torso_len) or torso_len <= 0:
        torso_len = 1.0

    pts = []
    valid = []
    for k in ids:
        kk = _joint_idx(V, k)
        if kk is None:
            continue
        pts.append(j[:, kk, :])
        valid.append(m[:, kk])
    if not pts:
        # No stable joints detected: treat as no detectable motion.
        return 0.0

    P = np.stack(pts, axis=1)  # [T,K,2]
    Vm = np.stack(valid, axis=1).astype(bool)  # [T,K]
    P_finite = np.isfinite(P).all(axis=2)
    Vm = Vm & P_finite

    # per-frame speed for each joint (masked)
    d = np.zeros_like(P, dtype=np.float32)
    d[1:] = P[1:] - P[:-1]
    speed = np.sqrt(np.sum(d * d, axis=2))  # [T,K]
    Vm_pair = Vm.copy()
    Vm_pair[0] = False
    Vm_pair[1:] = Vm[1:] & Vm[:-1]
    speed = np.where(Vm_pair, speed, np.nan)

    # Aggregate per frame safely:
    # frames with no valid joints default to 0.0 motion instead of NaN veto.
    frame_speed = _rowwise_nanmedian_or_default(speed, default=0.0)
    frame_speed = np.nan_to_num(frame_speed, nan=0.0, posinf=0.0, neginf=0.0)
    ms = _smooth_1d(frame_speed[sl], smooth)
    if not np.isfinite(ms):
        return 0.0

    # normalize and scale by fps to get per-second-ish
    ms = float(ms * fps / torso_len)
    return max(0.0, ms)


def confirm_scores_window(
    joints_xy: np.ndarray,
    mask: np.ndarray,
    fps: float,
    *,
    tail_s: float = 1.0,
    smooth: str = "median",
) -> Tuple[float, float]:
    """Compute `(lying_score, motion_score)` for one classified window.

    The pair is allowed to contain NaNs when the underlying pose evidence is
    unavailable. Downstream alert logic relies on that contract so missing
    heuristic signals remain ignorable rather than becoming accidental vetoes.
    """
    ls = lying_score_window(joints_xy, mask, fps, tail_s=tail_s, smooth=smooth)
    ms = motion_score_window(joints_xy, mask, fps, tail_s=tail_s, smooth=smooth)
    # Preserve "unavailable" signals as NaN so alert logic can ignore them
    # instead of treating them as hard vetoes.
    ls = _finite_or_default(ls, default=float("nan"))
    ms = _finite_or_default(ms, default=float("nan"))
    return ls, ms
