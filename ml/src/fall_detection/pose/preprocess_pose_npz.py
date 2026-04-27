#!/usr/bin/env python3
"""Clean and normalize pose NPZ files before temporal windowing.

This module owns sequence-level pose cleanup for extracted MediaPipe-style NPZ
files. It standardizes missing-value semantics, optionally fills and smooths
short gaps, applies body-centric normalization, and emits explicit validity
masks for downstream windowing/training code.

Why this rewrite?
-----------------
Your downstream windowing / models work best when the pose sequences:
  - have consistent "missing" semantics
  - have short gaps filled (optional)
  - are smoothed (optional)
  - are normalised in a body-centric coordinate system (optional)
  - include explicit validity masks (joint-level + frame-level)

Input NPZ (from your extractors)
--------------------------------
Required:
  - xy   : (T, 33, 2) float32  (often 0..1 image-normalised; may contain NaNs)
  - conf : (T, 33)    float32  (visibility/confidence; 0 means "missing")

Optional (carried through unchanged):
  - fps, size, src, seq_id, frames, etc.

Output NPZ
----------
Always writes:
  - xy, conf (same keys for backward compatibility)
  - mask        : (T, 33) uint8   1 if joint is valid (after fill, if enabled)
  - frame_mask  : (T,)    uint8   1 if frame has enough valid joints
  - valid_ratio : (T,)    float32 fraction of valid joints in frame
  - preprocess  : JSON string (parameters + summary)

Notes
-----
- We do NOT drop frames (to preserve alignment with span annotations).
  Instead, we provide frame_mask/valid_ratio and optionally zero-out
  frames that are too corrupted (see --invalidate_bad_frames).
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# MediaPipe Pose landmark indices used for normalisation / rotation
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24


# -------------------------
# IO helpers
# -------------------------
def list_npz(in_dir: str, recursive: bool) -> list[Path]:
    """List pose NPZ files from one directory tree in deterministic order."""

    root = Path(in_dir)
    return sorted(root.rglob("*.npz") if recursive else root.glob("*.npz"))


# -------------------------
# Missing-value handling
# -------------------------
def standardize_missing(xy: np.ndarray, conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make "missing" consistent:
      - non-finite conf -> 0
      - where conf <= 0 OR xy non-finite -> xy becomes NaN

    Downstream fill/smooth/normalize logic relies on NaN in ``xy`` as the single
    missing-value marker, so this step runs before every other transform.
    """
    xy2 = xy.astype(np.float32, copy=True)
    conf2 = conf.astype(np.float32, copy=True)

    conf2 = np.nan_to_num(conf2, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    bad_xy = ~np.isfinite(xy2[..., 0]) | ~np.isfinite(xy2[..., 1])
    bad_conf = conf2 <= 0.0
    miss = bad_xy | bad_conf  # (T,J)

    # Set missing xy to NaN (both channels)
    xy2[miss] = np.nan
    conf2[bad_xy] = 0.0
    return xy2, conf2


def clip_xy_finite(xy: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip finite xy coordinates to [lo, hi] while preserving NaN missing markers."""
    out = xy.astype(np.float32, copy=True)
    fin = np.isfinite(out)
    if not fin.any():
        return out
    lo_f = np.float32(lo)
    hi_f = np.float32(hi)
    out[fin] = np.clip(out[fin], lo_f, hi_f)
    return out


def limit_step_displacement(xy: np.ndarray, max_step: float, eps: float = 1e-6) -> np.ndarray:
    """Limit per-frame joint displacement magnitude to reduce extraction spikes.

    Applies a causal clamp:
      ||xy[t,j] - xy[t-1,j]|| <= max_step
    for finite coordinate pairs only. Missing values (NaN) are preserved.
    """
    ms = float(max_step)
    if ms <= 0.0:
        return xy
    ms32 = np.float32(ms)
    ms2 = ms32 * ms32
    eps2 = np.float32(eps * eps)
    out = xy.astype(np.float32, copy=True)
    if out.shape[0] < 2:
        return out

    # Causal clamp over time; vectorized across joints/channels per frame.
    for ti in range(1, out.shape[0]):
        prev = out[ti - 1]  # [J,2]
        cur = out[ti]       # [J,2]
        d = cur - prev
        valid = (
            np.isfinite(prev[:, 0]) & np.isfinite(prev[:, 1]) &
            np.isfinite(cur[:, 0]) & np.isfinite(cur[:, 1])
        )
        if not valid.any():
            continue
        dist2 = (d[:, 0] * d[:, 0]) + (d[:, 1] * d[:, 1])
        need = valid & (dist2 > ms2)
        if not need.any():
            continue
        scale = ms32 / np.sqrt(np.maximum(dist2[need], eps2))
        cur[need] = prev[need] + (d[need] * scale[:, None])
    return out


# -------------------------
# Gap filling (xy + conf)
# -------------------------
def linear_fill_small_gaps(
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
    max_gap: int,
    fill_conf: str = "thr",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill short missing gaps (<= max_gap) by linear interpolation for each joint.

    Parameters
    ----------
    fill_conf:
      - "keep"         : keep conf as-is (filled xy still has low conf)
      - "thr"          : set filled conf to conf_thr
      - "min_neighbors": set to min(conf[left], conf[right])
      - "linear"       : linear interpolation for conf too

    Returns
    -------
    xy_out, conf_out, filled_mask  where filled_mask is (T,J) bool.

    Only interior gaps bounded by valid observations are filled. Leading/trailing
    missing segments stay missing so the pipeline does not invent motion beyond
    observed evidence.
    """
    if max_gap <= 0:
        return xy, conf, np.zeros(conf.shape, dtype=bool)

    fill_conf = fill_conf.lower()
    if fill_conf not in ("keep", "thr", "min_neighbors", "linear"):
        raise ValueError("fill_conf must be one of: keep|thr|min_neighbors|linear")

    T, J, _ = xy.shape

    valid = (conf >= conf_thr) & np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])  # (T,J)
    # Common fast path: nothing missing above confidence threshold.
    if bool(np.all(valid)):
        return xy, conf, np.zeros((T, J), dtype=bool)

    # Reuse interpolation vectors across files/runs to reduce hot-path allocation churn.
    alpha_cache = _alpha_cache(max_gap)
    conf_thr_f = np.float32(conf_thr)
    mode_keep = (fill_conf == "keep")
    mode_thr = (fill_conf == "thr")
    mode_min_neighbors = (fill_conf == "min_neighbors")
    mode_linear = (fill_conf == "linear")
    need_conf_interp = mode_min_neighbors or mode_linear
    # Precompute per-joint validity once and iterate only joints that can have
    # fillable internal gaps (at least 2 valid points and at least 1 missing).
    valid_counts = valid.sum(axis=0, dtype=np.int32) if T > 0 else np.zeros((J,), dtype=np.int32)
    candidate_js = np.flatnonzero((valid_counts >= 2) & (valid_counts < T))
    if candidate_js.size == 0:
        return xy, conf, np.zeros((T, J), dtype=bool)

    out_xy = xy.copy()
    out_conf = conf.copy()
    filled = np.zeros((T, J), dtype=bool)
    any_filled = False

    for j in candidate_js:
        valid_j = valid[:, j]
        xj = out_xy[:, j, :]  # (T,2)
        conf_j = out_conf[:, j] if need_conf_interp else None
        filled_j = filled[:, j]
        vidx = np.flatnonzero(valid_j)
        if vidx.size < 2:
            continue
        left_idx = vidx[:-1]
        right_idx = vidx[1:]
        gaps = right_idx - left_idx - 1
        gap_ok = (gaps > 0) & (gaps <= max_gap)
        if not gap_ok.any():
            continue

        left_ok = left_idx[gap_ok]
        right_ok = right_idx[gap_ok]
        gaps_ok = gaps[gap_ok]

        # Batch by gap length (usually tiny, e.g. <=4), reducing per-gap Python work.
        for gap_len in range(1, max_gap + 1):
            use = gaps_ok == gap_len
            if not use.any():
                continue

            lefts = left_ok[use]
            rights = right_ok[use]
            rows = lefts[:, None] + np.arange(1, gap_len + 1, dtype=np.int32)[None, :]

            alpha = alpha_cache[gap_len][None, :, None]
            x_left = xj[lefts][:, None, :]
            x_right = xj[rights][:, None, :]
            xj[rows] = x_left + ((x_right - x_left) * alpha)
            filled_j[rows] = True
            any_filled = True

            if mode_thr or mode_keep:
                continue

            assert conf_j is not None
            if mode_min_neighbors:
                cfill = np.minimum(conf_j[lefts], conf_j[rights])
                np.maximum(cfill, conf_thr_f, out=cfill)
                conf_j[rows] = cfill[:, None]
            elif mode_linear:
                ca = alpha_cache[gap_len][None, :]
                cl = conf_j[lefts][:, None]
                cr = conf_j[rights][:, None]
                conf_j[rows] = cl + ((cr - cl) * ca)

    if mode_thr and any_filled:
        np.putmask(out_conf, filled & (out_conf < conf_thr_f), conf_thr_f)

    return out_xy, out_conf, filled


@lru_cache(maxsize=32)
def _alpha_cache(max_gap: int) -> Tuple[np.ndarray, ...]:
    mg = int(max_gap)
    if mg <= 0:
        return (np.asarray([], dtype=np.float32),)
    out = [np.asarray([], dtype=np.float32) for _ in range(mg + 1)]
    for g in range(1, mg + 1):
        out[g] = (np.arange(1, g + 1, dtype=np.float32) / np.float32(g + 1))
    return tuple(out)


# -------------------------
# Smoothing
# -------------------------
def smooth_weighted_moving_average(xy: np.ndarray, conf: np.ndarray, conf_thr: float, k: int) -> np.ndarray:
    """
    Weighted moving average smoothing.
    If a joint has no valid samples in the window, output remains NaN.
    """
    if k <= 1:
        return xy
    if k % 2 == 0:
        k += 1

    T, _J, _C = xy.shape
    out = np.full_like(xy, np.nan, dtype=np.float32)
    if T == 0:
        return out

    conf_arr = np.asarray(conf, dtype=np.float32)
    valid_conf = (conf_arr >= conf_thr) & np.isfinite(conf_arr)
    if bool(np.all(valid_conf)):
        w = conf_arr
    else:
        w = conf_arr.copy()
        w[~valid_conf] = 0.0
    half = k // 2

    # Manual edge padding is faster than np.pad(..., mode="edge") on this hot path.
    Tpad = T + (2 * half)
    xy_pad = np.empty((Tpad, xy.shape[1], 2), dtype=np.float32)
    w_pad = np.empty((Tpad, w.shape[1]), dtype=np.float32)
    xy_pad[half : half + T] = xy
    w_pad[half : half + T] = w
    if half > 0:
        xy_pad[:half] = xy[0]
        xy_pad[half + T :] = xy[-1]
        w_pad[:half] = w[0]
        w_pad[half + T :] = w[-1]

    # Vectorized rolling sums via cumulative sums; avoids [T,k,J,*] window tensors.
    # Fast path: all XY finite, so no validity mask materialization is needed.
    if bool(np.isfinite(xy).all()):
        w_eff = w_pad
        use_valid_where = False
        valid_xy = None
    else:
        valid_src = np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])  # (T,J)
        valid_xy = np.empty((Tpad, valid_src.shape[1]), dtype=bool)
        valid_xy[half : half + T] = valid_src
        if half > 0:
            valid_xy[:half] = valid_src[0]
            valid_xy[half + T :] = valid_src[-1]
        w_eff = w_pad.copy()
        w_eff[~valid_xy] = 0.0
        use_valid_where = True

    # Rolling sum via cumulative sums; avoids building [T+1,...] prefix arrays.
    # For csum over length L, k-window sums are:
    #   out[0]   = csum[k-1]
    #   out[i>0] = csum[i+k-1] - csum[i-1]
    csum_w = np.cumsum(w_eff, axis=0, dtype=np.float32)
    denom = csum_w[k - 1 :].copy()
    if k > 1:
        denom[1:] -= csum_w[:-k]
    ok = denom > 1e-8
    if not ok.any():
        return out

    weighted = np.empty_like(w_eff, dtype=np.float32)
    if use_valid_where:
        assert valid_xy is not None
        weighted.fill(0.0)
        np.multiply(xy_pad[..., 0], w_eff, out=weighted, where=valid_xy)
    else:
        np.multiply(xy_pad[..., 0], w_eff, out=weighted)
    csum_x = np.cumsum(weighted, axis=0, dtype=np.float32)
    num_x = csum_x[k - 1 :].copy()
    if k > 1:
        num_x[1:] -= csum_x[:-k]

    if use_valid_where:
        assert valid_xy is not None
        weighted.fill(0.0)
        np.multiply(xy_pad[..., 1], w_eff, out=weighted, where=valid_xy)
    else:
        np.multiply(xy_pad[..., 1], w_eff, out=weighted)
    csum_y = np.cumsum(weighted, axis=0, dtype=np.float32)
    num_y = csum_y[k - 1 :].copy()
    if k > 1:
        num_y[1:] -= csum_y[:-k]
    np.divide(num_x, denom, out=out[..., 0], where=ok)
    np.divide(num_y, denom, out=out[..., 1], where=ok)

    return out


# -------------------------
# Normalisation / rotation
# -------------------------
def _fill_nearest_2d(center: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid frames in center[T,2] with nearest valid frame (in time)."""
    T = center.shape[0]
    valid2 = valid & np.isfinite(center[:, 0]) & np.isfinite(center[:, 1])
    if valid2.sum() == 0:
        return np.zeros_like(center, dtype=np.float32)

    idx = np.arange(T)
    last_valid = np.where(valid2, idx, -1)
    last = np.maximum.accumulate(last_valid)

    next_valid = np.where(valid2, idx, T)
    nxt = np.minimum.accumulate(next_valid[::-1])[::-1]

    choose = np.where(
        last == -1,
        nxt,
        np.where(
            nxt == T,
            last,
            np.where((idx - last) <= (nxt - idx), last, nxt),
        ),
    ).astype(np.int64, copy=False)

    out = center.copy().astype(np.float32)
    bad = ~valid2
    out[bad] = center[choose[bad]]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out



def _fill_forward_2d(center: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Fill invalid frames in center[T,2] using last valid (forward fill).

    Notes
    -----
    - This is causal after the first valid frame appears.
    - For leading invalid frames (before the first valid), we backfill with the first valid
      to avoid large artificial translation from zero-centering.
    """
    T = center.shape[0]
    valid2 = valid & np.isfinite(center[:, 0]) & np.isfinite(center[:, 1])
    if valid2.sum() == 0:
        return np.zeros_like(center, dtype=np.float32)

    idx = np.arange(T)
    first = int(idx[valid2][0])
    last_valid = np.where(valid2, idx, -1)
    last = np.maximum.accumulate(last_valid)
    fill_idx = np.maximum(last, first).astype(np.int64, copy=False)
    out = center[fill_idx].astype(np.float32, copy=False)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def _joint_center(xy: np.ndarray, conf: np.ndarray, a: int, b: int, conf_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return center[T,2] and valid[T] using joints a/b with fallback if one missing."""
    T = xy.shape[0]
    J = xy.shape[1] if xy.ndim >= 2 else 0
    if a < 0 or b < 0 or a >= J or b >= J:
        return np.zeros((T, 2), dtype=np.float32), np.zeros((T,), dtype=bool)
    pa, pb = xy[:, a, :], xy[:, b, :]
    va = conf[:, a] >= conf_thr
    vb = conf[:, b] >= conf_thr

    center = np.full((T, 2), np.nan, np.float32)
    both = va & vb
    only_a = va & ~vb
    only_b = vb & ~va
    center[both] = 0.5 * (pa[both] + pb[both])
    center[only_a] = pa[only_a]
    center[only_b] = pb[only_b]
    valid = both | only_a | only_b
    return center, valid


def _rotation_angle_shoulders(xy: np.ndarray, conf: np.ndarray, conf_thr: float) -> float:
    """
    Compute a stable (sequence-level) rotation angle using shoulder vector.
    We rotate by -angle so that the shoulder line becomes horizontal.
    """
    if xy.shape[1] <= max(L_SHO, R_SHO):
        return 0.0
    ls = xy[:, L_SHO, :]
    rs = xy[:, R_SHO, :]
    ok = (conf[:, L_SHO] >= conf_thr) & (conf[:, R_SHO] >= conf_thr)
    ok = ok & np.isfinite(ls[:, 0]) & np.isfinite(ls[:, 1]) & np.isfinite(rs[:, 0]) & np.isfinite(rs[:, 1])
    if ok.sum() == 0:
        return 0.0
    v = rs[ok] - ls[ok]
    ang = np.arctan2(v[:, 1], v[:, 0])  # radians
    # Use median to reduce jitter/outliers
    a = float(np.median(ang))
    if not np.isfinite(a):
        return 0.0
    return a


def normalize_body_centric(
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
    mode: str,
    pelvis_fill: str = "nearest",
    rotate: str = "none",
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Translate to pelvis center and scale by torso length or shoulder width.
    Optional: rotate so shoulders are horizontal (sequence-level).

    The transform is sequence-level, not per-frame adaptive scaling. That keeps a
    window internally consistent for downstream temporal models.
    """
    mode = mode.lower()
    rotate = rotate.lower()
    if mode == "none" and rotate == "none":
        return xy, {"norm": "none", "rot": "none"}

    if rotate not in ("none", "shoulders"):
        raise ValueError("rotate must be one of: none|shoulders")

    T = xy.shape[0]
    J = xy.shape[1] if xy.ndim >= 2 else 0
    if T == 0 or J == 0:
        return xy.astype(np.float32, copy=False), {"norm": mode, "rot": rotate, "scale": 1.0, "rot_angle": 0.0}

    pelvis, pelvis_ok = _joint_center(xy, conf, L_HIP, R_HIP, conf_thr)
    shoulders, sh_ok = _joint_center(xy, conf, L_SHO, R_SHO, conf_thr)

    if mode == "none":
        scale = 1.0
    elif mode == "torso":
        d = np.linalg.norm(shoulders - pelvis, axis=1)
        valid_d = pelvis_ok & sh_ok & np.isfinite(d)
        scale = float(np.median(d[valid_d])) if valid_d.sum() else 1.0
    elif mode == "shoulder":
        if J <= max(L_SHO, R_SHO):
            scale = 1.0
        else:
            d = np.linalg.norm(xy[:, L_SHO, :] - xy[:, R_SHO, :], axis=1)
            valid_d = (conf[:, L_SHO] >= conf_thr) & (conf[:, R_SHO] >= conf_thr) & np.isfinite(d)
            scale = float(np.median(d[valid_d])) if valid_d.sum() else 1.0
    else:
        raise ValueError(f"Unknown normalize mode: {mode} (use none|torso|shoulder)")

    if (not np.isfinite(scale)) or scale < eps:
        scale = 1.0

    if pelvis_fill == "nearest":
        pelvis_used = _fill_nearest_2d(pelvis, pelvis_ok)
    elif pelvis_fill == "forward":
        pelvis_used = _fill_forward_2d(pelvis, pelvis_ok)
    else:
        # "zero": keep valid pelvis, replace missing with (0,0)
        pelvis_used = np.nan_to_num(pelvis, nan=0.0, posinf=0.0, neginf=0.0)

    xy2 = xy.astype(np.float32, copy=False)

    # Translate and scale
    if mode != "none":
        xy2 = (xy2 - pelvis_used[:, None, :]) / float(scale)
    else:
        xy2 = (xy2 - pelvis_used[:, None, :])

    # Rotate around origin (already pelvis-centered)
    rot_angle = 0.0
    if rotate == "shoulders":
        rot_angle = _rotation_angle_shoulders(xy, conf, conf_thr)
        ca = float(np.cos(-rot_angle))
        sa = float(np.sin(-rot_angle))
        R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
        xy2 = xy2 @ R.T

    meta = {
        "norm": mode,
        "scale": float(scale),
        "rot": rotate,
        "rot_angle": float(rot_angle),
        "conf_thr": float(conf_thr),
    }
    return xy2.astype(np.float32), meta


# -------------------------
# Masks / frame quality
# -------------------------
def compute_masks(xy: np.ndarray, conf: np.ndarray, conf_thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      joint_mask  : (T,J) bool
      frame_mask  : (T,)  bool  (refined by min_valid_ratio in process_one)
      valid_ratio : (T,)  float32

    ``frame_mask`` intentionally starts permissive here; ``process_one`` applies
    dataset/task-specific frame-quality thresholds later.
    """
    joint_mask = (conf >= conf_thr) & np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])
    n_joints = int(joint_mask.shape[1]) if joint_mask.ndim == 2 else 0
    denom = float(max(1, n_joints))
    valid_ratio = (joint_mask.sum(axis=1, dtype=np.int32).astype(np.float32) / denom) if joint_mask.ndim == 2 else np.asarray([], dtype=np.float32)
    frame_mask = np.isfinite(valid_ratio)  # refined later
    return joint_mask, frame_mask, valid_ratio


def invalidate_bad_frames(
    xy: np.ndarray,
    conf: np.ndarray,
    joint_mask: np.ndarray,
    frame_mask: np.ndarray,
    valid_ratio: np.ndarray,
    min_valid_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    If a frame has too few valid joints, mark it invalid and zero-out xy/conf.
    This preserves indexing while preventing bad frames from harming downstream.
    """
    if min_valid_ratio <= 0.0:
        return xy, conf, frame_mask

    fm = frame_mask.copy()
    bad = valid_ratio < float(min_valid_ratio)
    if not bad.any():
        return xy, conf, fm

    fm[bad] = False
    xy2 = xy.copy()
    conf2 = conf.copy()

    xy2[bad, :, :] = 0.0
    conf2[bad, :] = 0.0
    joint_mask[bad, :] = False
    return xy2, conf2, fm


# -------------------------
# Main per-file processing
# -------------------------
def process_one(in_path: Path, out_path: Path, args) -> bool:
    """Process one pose NPZ and write the normalized output NPZ.

    The processing order is fixed: standardize missing -> optional clipping ->
    short-gap fill -> optional step clamp -> smoothing -> normalization ->
    mask/frame-quality derivation -> optional frame invalidation. The function
    preserves frame count and carries through unrelated metadata keys.
    """

    if args.skip_existing and out_path.exists():
        return True

    with np.load(in_path, allow_pickle=False) as d:
        if "xy" not in d.files or "conf" not in d.files:
            return False

        xy = d["xy"].astype(np.float32)
        conf = d["conf"].astype(np.float32)

        # Preserve unrelated extractor metadata so later stages do not lose file
        # provenance or sequence information when preprocessing is introduced.
        extras = {k: d[k] for k in d.files if k not in ("xy", "conf", "preprocess", "mask", "frame_mask", "valid_ratio")}

    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"{in_path}: expected xy (T,J,2), got {xy.shape}")
    if conf.ndim != 2:
        raise ValueError(f"{in_path}: expected conf (T,J), got {conf.shape}")
    if xy.shape[0] != conf.shape[0] or xy.shape[1] != conf.shape[1]:
        raise ValueError(f"{in_path}: xy {xy.shape} and conf {conf.shape} mismatch")

    # 1) standardize missing semantics
    xy, conf = standardize_missing(xy, conf)
    if args.clip_xy:
        xy = clip_xy_finite(xy, args.clip_xy_min, args.clip_xy_max)

    # 2) fill short gaps
    filled_mask = np.zeros(conf.shape, dtype=bool)
    if args.max_gap > 0:
        xy, conf, filled_mask = linear_fill_small_gaps(
            xy, conf, conf_thr=args.conf_thr, max_gap=args.max_gap, fill_conf=args.fill_conf
        )
    if args.max_step > 0.0:
        xy = limit_step_displacement(xy, args.max_step)

    # 3) smooth
    if args.smooth_k > 1:
        xy = smooth_weighted_moving_average(xy, conf, conf_thr=args.conf_thr, k=args.smooth_k)

    # 4) normalise + optional rotation
    xy, norm_meta = normalize_body_centric(
        xy,
        conf,
        conf_thr=args.conf_thr,
        mode=args.normalize,
        pelvis_fill=args.pelvis_fill,
        rotate=args.rotate,
    )

    # 5) masks / frame quality
    joint_mask, frame_mask, valid_ratio = compute_masks(xy, conf, conf_thr=args.conf_thr)

    # frame_mask carries the quality contract even when we choose not to zero-out
    # bad frames, so downstream code can decide whether to ignore them.
    if args.min_valid_ratio > 0.0:
        frame_mask = frame_mask & (valid_ratio >= float(args.min_valid_ratio))

    if args.invalidate_bad_frames:
        xy, conf, frame_mask = invalidate_bad_frames(
            xy, conf, joint_mask, frame_mask, valid_ratio, min_valid_ratio=args.min_valid_ratio
        )

    # Final: replace NaNs with 0 for safe downstream IO
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Persist preprocessing choices alongside the cleaned arrays so later runs
    # can audit exactly which transforms produced this NPZ.
    preprocess_meta = dict(
        conf_thr=float(args.conf_thr),
        smooth_k=int(args.smooth_k),
        max_gap=int(args.max_gap),
        fill_conf=str(args.fill_conf),
        max_step=float(args.max_step),
        normalize=str(args.normalize),
        rotate=str(args.rotate),
        pelvis_fill=str(args.pelvis_fill),
        min_valid_ratio=float(args.min_valid_ratio),
        invalidate_bad_frames=bool(args.invalidate_bad_frames),
        clip_xy=bool(args.clip_xy),
        clip_xy_min=float(args.clip_xy_min),
        clip_xy_max=float(args.clip_xy_max),
        frame_mask_min_valid_ratio_applied=True,
        frames=int(xy.shape[0]),
        joints=int(xy.shape[1]),
        filled_points=int(filled_mask.sum()),
        valid_ratio_mean=float(np.nanmean(valid_ratio)) if valid_ratio.size else 0.0,
        valid_ratio_min=float(np.nanmin(valid_ratio)) if valid_ratio.size else 0.0,
        **{k: v for k, v in norm_meta.items() if k != "conf_thr"},
    )

    np.savez_compressed(
        out_path,
        xy=xy,
        conf=conf,
        mask=joint_mask.astype(np.uint8),
        frame_mask=frame_mask.astype(np.uint8),
        valid_ratio=valid_ratio.astype(np.float32),
        preprocess=json.dumps(preprocess_meta),
        **extras,
    )
    return True


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Clean + normalise pose NPZs before windowing.")
    ap.add_argument("--in_dir", required=True, help="Directory containing pose NPZ files.")
    ap.add_argument("--out_dir", required=True, help="Where to write cleaned pose NPZ files.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    ap.add_argument("--skip_existing", action="store_true", help="Skip writing if output already exists.")

    ap.add_argument("--conf_thr", type=float, default=0.2, help="Confidence threshold for 'valid' joints.")
    ap.add_argument("--smooth_k", type=int, default=5, help="Weighted moving-average window size (odd preferred).")

    ap.add_argument("--max_gap", type=int, default=4, help="Max missing gap (frames) to interpolate.")
    ap.add_argument(
        "--fill_conf",
        choices=["keep", "thr", "min_neighbors", "linear"],
        default="thr",
        help="How to set conf for interpolated points (default: thr).",
    )
    ap.add_argument(
        "--max_step",
        type=float,
        default=0.0,
        help="Optional max per-frame joint displacement (normalized units); 0 disables.",
    )

    ap.add_argument(
        "--normalize",
        choices=["none", "torso", "shoulder"],
        default="torso",
        help="Body-centric normalization mode.",
    )
    ap.add_argument(
        "--rotate",
        choices=["none", "shoulders"],
        default="none",
        help="Optional rotation: align shoulders horizontally (sequence-level).",
    )
    ap.add_argument(
        "--pelvis_fill",
        choices=["nearest", "forward", "zero"],
        default="nearest",
        help="How to fill missing pelvis center before translation (nearest is non-causal; forward is causal after first valid).",
    )

    ap.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.25,
        help="Minimum fraction of valid joints for a frame to be considered usable.",
    )
    ap.add_argument(
        "--invalidate_bad_frames",
        action="store_true",
        help="If set, frames with valid_ratio < min_valid_ratio are zeroed out + flagged in frame_mask.",
    )
    ap.add_argument(
        "--clip_xy",
        action="store_true",
        help="If set, clip finite xy coordinates to [clip_xy_min, clip_xy_max] before fill/smooth/normalize.",
    )
    ap.add_argument("--clip_xy_min", type=float, default=0.0, help="Lower bound used with --clip_xy.")
    ap.add_argument("--clip_xy_max", type=float, default=1.0, help="Upper bound used with --clip_xy.")

    ap.add_argument("--log_every", type=int, default=200, help="Print progress every N files (default 200).")
    return ap.parse_args()


def main():
    args = parse_args()
    files = list_npz(args.in_dir, args.recursive)
    if not files:
        raise SystemExit(f"[ERR] no .npz files under: {args.in_dir}")

    in_root = Path(args.in_dir).resolve()
    out_root = Path(args.out_dir).resolve()

    ok, fail = 0, 0
    for i, p in enumerate(files, start=1):
        rel = p.resolve().relative_to(in_root)
        out_p = out_root / rel

        try:
            if process_one(p, out_p, args):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"[ERR] {p}: {e}")
            fail += 1

        if args.log_every > 0 and (i % args.log_every == 0):
            print(f"[prog] {i}/{len(files)}  ok={ok}  fail={fail}")

    print(f"[done] wrote {ok} cleaned files to {out_root}  (failed/skipped={fail})")


if __name__ == "__main__":
    main()
