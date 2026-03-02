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

_MP_SHOULDERS = [11, 12]
_MP_HIPS = [23, 24]
_MP_STABLE_IDS = [11, 12, 23, 24, 27, 28]
_COCO_SHOULDERS = [5, 6]
_COCO_HIPS = [11, 12]
_COCO_STABLE_IDS = [5, 6, 11, 12, 15, 16]


@dataclass(frozen=True)
class ConfirmCfg:
    tail_s: float = 1.0  # use last ~1s
    smooth: str = "median"  # median|mean


def _safe_fps(fps: float, *, default: float = 30.0) -> float:
    try:
        f = float(fps)
    except Exception:
        f = default
    if not np.isfinite(f) or f <= 0:
        f = default
    return f


def _tail_n(T: int, fps: float, tail_s: float) -> int:
    n = int(np.ceil(_safe_fps(fps) * float(tail_s)))
    n = max(1, min(int(T), n))
    return n


def _smooth_1d(x: np.ndarray, how: str) -> float:
    if isinstance(x, np.ndarray):
        arr = x.reshape(-1)
        if arr.dtype not in (np.float32, np.float64):
            arr = arr.astype(np.float32, copy=False)
    else:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
    # Common fast path in production: fully finite arrays.
    if bool(np.isfinite(arr).all()):
        x_fin = arr
    else:
        fin = np.isfinite(arr)
        if not fin.any():
            return float("nan")
        x_fin = arr[fin]
    how = str(how).lower()
    if how == "mean":
        return float(np.mean(x_fin))
    return float(np.median(x_fin))


def _bbox_hw(
    j: np.ndarray,
    m: np.ndarray,
    *,
    check_finite: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame bbox height/width from visible joints.

    Safe against all-NaN frames (no warnings); returns NaNs for those frames.
    """
    # j: [T,V,2], m: [T,V]
    T, V, _ = j.shape
    m = np.asarray(m, dtype=bool)
    if check_finite:
        m = m & np.isfinite(j[..., 0]) & np.isfinite(j[..., 1])
    valid = m.any(axis=1)  # [T]
    if not bool(valid.any()):
        nan_arr = np.full((T,), np.nan, dtype=np.float32)
        return nan_arr.copy(), nan_arr

    x = j[..., 0]
    y = j[..., 1]
    # Common fast path: all joints valid in all frames.
    if valid.all():
        xmin = np.min(x, axis=1)
        xmax = np.max(x, axis=1)
        ymin = np.min(y, axis=1)
        ymax = np.max(y, axis=1)
    else:
        # Use +/-inf masking and regular min/max to avoid nanmin/nanmax overhead.
        xmin = np.min(np.where(m, x, np.float32(np.inf)), axis=1)
        xmax = np.max(np.where(m, x, np.float32(-np.inf)), axis=1)
        ymin = np.min(np.where(m, y, np.float32(np.inf)), axis=1)
        ymax = np.max(np.where(m, y, np.float32(-np.inf)), axis=1)

    h = (ymax - ymin).astype(np.float32, copy=False)
    w = (xmax - xmin).astype(np.float32, copy=False)
    if not valid.all():
        bad = ~valid
        h[bad] = np.nan
        w[bad] = np.nan
    return h, w


def _joint_idx(V: int, idx: int) -> Optional[int]:
    return int(idx) if 0 <= int(idx) < int(V) else None


def _mean_of(
    j: np.ndarray,
    m: np.ndarray,
    ids: Iterable[int],
    *,
    assume_finite_mask: bool = False,
    assume_ids_valid: bool = False,
) -> np.ndarray:
    """Compute per-frame mean of given joints; returns NaNs if none valid.

    Avoids RuntimeWarning for frames where all joints are missing.
    """
    T, V, _ = j.shape
    if assume_ids_valid:
        valid_ids = [int(k) for k in ids]
    else:
        valid_ids = [int(k) for k in ids if _joint_idx(V, k) is not None]
    if not valid_ids:
        return np.full((T, 2), np.nan, dtype=np.float32)

    P = j[:, valid_ids, :]  # [T,K,2]
    if isinstance(m, np.ndarray) and m.dtype == np.bool_:
        Mm = m[:, valid_ids]
    else:
        Mm = np.asarray(m[:, valid_ids], dtype=bool)
    if not assume_finite_mask:
        Mm = Mm & np.isfinite(P[..., 0]) & np.isfinite(P[..., 1])

    out = np.full((T, 2), np.nan, dtype=np.float32)
    cnt = Mm.sum(axis=1).astype(np.float32, copy=False)  # [T]
    valid = cnt > 0
    if valid.any():
        Mm_f = Mm.astype(np.float32, copy=False)
        weighted = np.empty_like(P, dtype=np.float32)
        np.multiply(P, Mm_f[..., None], out=weighted)
        num = np.sum(weighted, axis=1, dtype=np.float32)
        out[valid] = num[valid] / cnt[valid, None]
    return out


def _frame_speed_from_valid_pairs(P: np.ndarray, Vm: np.ndarray) -> np.ndarray:
    """Return per-frame median joint speed with NaN-safe masking.

    P: [T,K,2] joint coordinates
    Vm: [T,K] boolean validity mask
    """
    T = int(P.shape[0])
    d = np.empty_like(P, dtype=np.float32)
    d[0] = 0.0
    np.subtract(P[1:], P[:-1], out=d[1:])
    speed = np.hypot(d[..., 0], d[..., 1])  # [T,K]
    Vm_pair = np.empty_like(Vm, dtype=bool)
    Vm_pair[0] = False
    np.logical_and(Vm[1:], Vm[:-1], out=Vm_pair[1:])
    if bool(Vm_pair.all()):
        return np.median(speed, axis=1).astype(np.float32, copy=False)
    speed[~Vm_pair] = np.nan
    frame_speed = np.full((T,), np.nan, dtype=np.float32)
    valid_f = Vm_pair.any(axis=1)
    if valid_f.any():
        frame_speed[valid_f] = np.nanmedian(speed[valid_f], axis=1).astype(np.float32)
    return frame_speed


def lying_score_window(
    joints_xy: np.ndarray,
    mask: np.ndarray,
    fps: float,
    *,
    tail_s: float = 1.0,
    smooth: str = "median",
) -> float:
    """Window-level lying score in [0,1]."""
    lying, _motion = confirm_scores_window(
        joints_xy,
        mask,
        fps,
        tail_s=tail_s,
        smooth=smooth,
    )
    return float(lying)


def motion_score_window(
    joints_xy: np.ndarray,
    mask: np.ndarray,
    fps: float,
    *,
    tail_s: float = 1.0,
    smooth: str = "median",
) -> float:
    """Window-level motion score (smaller = more still)."""
    _lying, motion = confirm_scores_window(
        joints_xy,
        mask,
        fps,
        tail_s=tail_s,
        smooth=smooth,
    )
    return float(motion)


def confirm_scores_window(
    joints_xy: np.ndarray,
    mask: np.ndarray,
    fps: float,
    *,
    tail_s: float = 1.0,
    smooth: str = "median",
) -> Tuple[float, float]:
    """Compute (lying_score, motion_score) for a single window with shared intermediates."""
    j = np.asarray(joints_xy, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if j.ndim != 3 or j.shape[-1] != 2:
        raise ValueError("joints_xy must be [T,V,2]")

    T, V, _ = j.shape
    m = m & np.isfinite(j[..., 0]) & np.isfinite(j[..., 1])
    if not m.any():
        return 0.0, float("inf")

    fps_safe = _safe_fps(fps)
    n_tail = _tail_n(T, fps_safe, tail_s)
    sl = slice(T - n_tail, T)

    if V >= 33:
        torso_idx = [11, 12, 23, 24]
        P4 = j[:, torso_idx, :]  # [T,4,2]
        M4 = m[:, torso_idx]  # [T,4]
        sh = np.full((T, 2), np.nan, dtype=np.float32)
        hip = np.full((T, 2), np.nan, dtype=np.float32)

        M_sh = M4[:, :2]
        cnt_sh = M_sh.sum(axis=1).astype(np.float32, copy=False)
        ok_sh = cnt_sh > 0.0
        if bool(np.any(ok_sh)):
            w_sh = M_sh.astype(np.float32, copy=False)
            num_sh = np.sum(P4[:, :2, :] * w_sh[..., None], axis=1, dtype=np.float32)
            sh[ok_sh] = num_sh[ok_sh] / cnt_sh[ok_sh, None]

        M_hip = M4[:, 2:]
        cnt_hip = M_hip.sum(axis=1).astype(np.float32, copy=False)
        ok_hip = cnt_hip > 0.0
        if bool(np.any(ok_hip)):
            w_hip = M_hip.astype(np.float32, copy=False)
            num_hip = np.sum(P4[:, 2:, :] * w_hip[..., None], axis=1, dtype=np.float32)
            hip[ok_hip] = num_hip[ok_hip] / cnt_hip[ok_hip, None]
        ids = _MP_STABLE_IDS
    else:
        sh = _mean_of(j, m, _COCO_SHOULDERS, assume_finite_mask=True)
        hip = _mean_of(j, m, _COCO_HIPS, assume_finite_mask=True)
        ids = _COCO_STABLE_IDS

    torso = sh - hip
    dy = np.abs(torso[:, 1])
    torso_norm = np.hypot(torso[:, 0], torso[:, 1]) + 1e-6
    torso_h = 1.0 - np.clip(dy / torso_norm, 0.0, 1.0)

    h, w = _bbox_hw(j, m, check_finite=False)
    ratio = h / (w + 1e-6)
    bbox_h = np.clip((1.5 - ratio) / (1.5 - 0.6), 0.0, 1.0)
    lying_frame = 0.6 * torso_h + 0.4 * bbox_h
    ls = _smooth_1d(lying_frame[sl], smooth)
    lying = float(np.clip(ls, 0.0, 1.0)) if np.isfinite(ls) else 0.0

    torso_len = np.nanmedian(torso_norm[sl])
    if not np.isfinite(torso_len) or torso_len <= 0:
        torso_len = 1.0

    if V >= 33:
        valid_ids = ids
    else:
        valid_ids = [int(k) for k in ids if _joint_idx(V, k) is not None]
    if not valid_ids:
        return lying, float("inf")

    P = j[:, valid_ids, :]
    Vm = m[:, valid_ids].astype(bool, copy=False)
    frame_speed = _frame_speed_from_valid_pairs(P, Vm)
    ms = _smooth_1d(frame_speed[sl], smooth)
    if not np.isfinite(ms):
        return lying, float("inf")
    motion = max(0.0, float(ms * fps_safe / torso_len))
    return lying, motion
