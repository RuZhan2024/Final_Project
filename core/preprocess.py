#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/preprocess.py

Shared preprocessing for skeleton (2D pose) sequences.

This module is designed to be:
- NumPy-only (so it can be imported by offline scripts AND server runtime later)
- Deterministic (same input -> same output)
- Robust to missing keypoints (hips/shoulders missing should NOT break normalization)

Pipeline overview (pose_npz -> proc_npz)
----------------------------------------
Input arrays:
  xy   : [T, J, 2] float32   (x,y in pixels or normalized coords depending on extractor)
  conf : [T, J]    float32   (keypoint confidence in [0,1])

Steps:
1) Standardize "missing" semantics:
   - conf NaN/inf -> 0
   - if conf <= 0 or xy non-finite => xy becomes NaN (explicit missing)

2) Resample to target/deploy FPS:
   - xy   : linear interpolation (vectorized)
   - conf : nearest neighbor (keeps "validity" more stable)

3) One-Euro smoothing (optional):
   - smooth xy in time to reduce jitter BEFORE computing velocity features

4) Normalize (optional):
   - pelvis center (hip midpoint)
   - scale by torso length (shoulder_mid - pelvis)
   - optional rotate so shoulder line becomes horizontal

IMPORTANT FIX in this version:
- pelvis/torso/rotation are computed with confidence-aware fallbacks,
  so frames with missing hips/shoulders do not distort the whole body.

5) Mask generation:
   mask[t, j] = 1 if xy finite AND conf >= conf_gate else 0

Output arrays (proc_npz schema):
  joints : [T, J, 2] float32  (normalized xy if normalize=True else smoothed xy)
  conf   : [T, J]    float32  (resampled conf)
  mask   : [T, J]    uint8    (0/1)
  meta   : JSON string (cfg + stats)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ============================================================
# MediaPipe landmark indices (stable across MediaPipe Pose)
# ============================================================
# Shoulders
L_SHO, R_SHO = 11, 12
# Hips
L_HIP, R_HIP = 23, 24


# ============================================================
# Config dataclasses
# ============================================================

@dataclass(frozen=True)
class OneEuroCfg:
    """
    One-Euro filter configuration.

    min_cutoff:
      Base cutoff frequency (Hz). Higher -> less smoothing (more responsive).
    beta:
      Speed coefficient. Higher -> less smoothing when motion is fast.
    d_cutoff:
      Cutoff for derivative filtering (Hz).
    """
    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreprocessCfg:
    """
    Preprocess configuration.

    target_fps:
      Resample everything to this FPS so offline matches deploy timebase.

    conf_gate:
      Confidence threshold for mask. If conf < conf_gate => keypoint treated as invalid.

    normalize:
      If True, do pelvis centering + torso-length scaling (+ optional rotation).

    rotate_shoulders:
      If True, rotate each frame so the shoulder line becomes horizontal.
      (Useful if camera angle varies; can help model invariance)

    one_euro:
      If True, apply One-Euro smoothing before normalization.

    one_euro_cfg:
      Nested One-Euro parameters.
    """
    target_fps: float = 30.0
    conf_gate: float = 0.20
    normalize: bool = True
    rotate_shoulders: bool = False
    one_euro: bool = True
    one_euro_cfg: OneEuroCfg = OneEuroCfg()

    def to_dict(self) -> Dict[str, Any]:
        # asdict() also expands nested dataclasses.
        return asdict(self)


# ============================================================
# Small helpers
# ============================================================

def _as_str(x: Any) -> str:
    """
    Convert various np scalar/string forms into a clean Python string.

    Why:
    - NPZ sometimes stores strings as 0-d arrays or bytes
    - str(np.array(...)) can produce "b'...'" which is annoying
    """
    try:
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        if isinstance(x, np.ndarray):
            if x.shape == ():
                return _as_str(x.item())
            if x.size == 1:
                return _as_str(x.reshape(-1)[0].item())
        return str(x)
    except Exception:
        return str(x)


def _safe_scalar(z: np.lib.npyio.NpzFile, key: str, default: float) -> float:
    """
    Read a scalar float from an NPZ safely.

    If key missing or malformed => return default.
    """
    if key not in z.files:
        return float(default)
    try:
        return float(np.array(z[key]).reshape(-1)[0])
    except Exception:
        return float(default)


def _finite_xy_mask(xy: np.ndarray) -> np.ndarray:
    """Return [T,J] boolean mask where both x and y are finite."""
    xy = np.asarray(xy)
    return np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])


# ============================================================
# 1) Standardize missing semantics
# ============================================================

def standardize_missing(xy: np.ndarray, conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make missing-data semantics consistent.

    Rules:
    - conf NaN/inf -> 0
    - if conf <= 0 OR xy non-finite -> xy becomes NaN (explicit missing)
    - if xy non-finite -> conf becomes 0

    Why:
    - Some extractors output (0,0) with low conf, others output NaN, etc.
    - Downstream logic becomes much simpler if "missing" is consistent.
    """
    xy2 = np.asarray(xy, dtype=np.float32).copy()
    conf2 = np.asarray(conf, dtype=np.float32).copy()

    # Replace NaN/inf confidence with 0 so it becomes "invalid"
    conf2 = np.nan_to_num(conf2, nan=0.0, posinf=0.0, neginf=0.0)

    # Identify missing due to coordinates or confidence
    bad_xy = ~_finite_xy_mask(xy2)   # xy is NaN/inf
    bad_conf = conf2 <= 0.0          # conf says keypoint is unreliable/missing

    missing = bad_xy | bad_conf

    # Enforce NaN for missing xy so we can see missing explicitly
    xy2[missing] = np.nan
    # If xy is invalid, conf must be 0
    conf2[bad_xy] = 0.0

    return xy2, conf2


# ============================================================
# 2) Resample to target FPS
# ============================================================

def resample_to_fps(
    xy: np.ndarray,
    conf: np.ndarray,
    fps_src: float,
    fps_tgt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a sequence from fps_src to fps_tgt.

    - xy: linear interpolation (vectorized)
    - conf: nearest-neighbor sampling

    Why nearest for conf:
    - conf acts like "validity", and nearest preserves sharp validity transitions
      better than linear interpolation.

    Duration preservation:
    - We preserve the duration based on (T-1)/fps_src and end-align the last frame.
    """
    xy = np.asarray(xy, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)

    T = int(xy.shape[0])
    if T <= 1:
        return xy.copy(), conf.copy()

    fs = float(fps_src) if float(fps_src) > 0 else float(fps_tgt)
    ft = float(fps_tgt) if float(fps_tgt) > 0 else float(fps_src)

    # If FPS is invalid or same -> return copy
    if fs <= 0 or ft <= 0 or abs(fs - ft) < 1e-6:
        return xy.copy(), conf.copy()

    # Duration in seconds for sequence endpoints
    dur_s = float((T - 1) / fs)

    # Target length chosen so that last target time ~= dur_s
    Tt = int(round(dur_s * ft)) + 1
    Tt = max(1, Tt)

    # Time grids (seconds)
    t_tgt = np.arange(Tt, dtype=np.float32) / ft

    # Convert target times to fractional indices in source frame space
    idx_f = t_tgt * fs  # float indices in [0, T-1]
    i0 = np.floor(idx_f).astype(np.int32)
    i0 = np.clip(i0, 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)

    # Interpolation factor in [0,1]
    a = (idx_f - i0).astype(np.float32)  # [Tt]
    a3 = a[:, None, None]                # broadcast to [Tt,1,1]

    # Grab source frames
    xy0 = xy[i0]
    xy1 = xy[i1]

    # IMPORTANT:
    # xy may contain NaNs for missing points.
    # If we linearly interpolate NaNs, NaNs spread everywhere.
    # We replace NaN with 0 here, but validity is still controlled by conf + mask later.
    xy0 = np.nan_to_num(xy0, nan=0.0, posinf=0.0, neginf=0.0)
    xy1 = np.nan_to_num(xy1, nan=0.0, posinf=0.0, neginf=0.0)

    xy_tgt = (1.0 - a3) * xy0 + a3 * xy1

    # Nearest-neighbor sampling for confidence
    idx_nn = np.clip(np.round(idx_f).astype(np.int32), 0, T - 1)
    conf_tgt = conf[idx_nn]

    return xy_tgt.astype(np.float32), conf_tgt.astype(np.float32)


# ============================================================
# 3) One-Euro filter
# ============================================================

def _alpha(cutoff_hz: np.ndarray, dt: float) -> np.ndarray:
    """
    Convert cutoff frequency to smoothing factor alpha.

    tau = 1 / (2*pi*cutoff)
    alpha = 1 / (1 + tau/dt)

    Larger cutoff => larger alpha => more responsive (less smoothing).
    """
    cutoff_hz = np.asarray(cutoff_hz, dtype=np.float32)
    cutoff_hz = np.maximum(cutoff_hz, 1e-6)
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


def one_euro_filter_xy(xy: np.ndarray, fps: float, cfg: OneEuroCfg) -> np.ndarray:
    """
    Apply One-Euro filter to xy over time.

    xy: [T, J, 2]

    Implementation details:
    - We flatten [J,2] into a single D dimension so the loop is simple.
    - This filters each coordinate independently.
    """
    x = np.asarray(xy, dtype=np.float32)
    T = int(x.shape[0])
    if T <= 2:
        return x.copy()

    fs = float(fps) if float(fps) > 0 else 30.0
    dt = 1.0 / fs

    # Flatten joints and xy dims: [T, D]
    D = int(x.shape[1] * x.shape[2])
    x2 = x.reshape(T, D)

    # Ensure no NaN/inf enters the filter
    x2 = np.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)

    # Derivative low-pass filtered estimate (dx_hat)
    dx_hat = np.zeros((D,), dtype=np.float32)

    # Filtered signal estimate (x_hat)
    x_hat = x2[0].copy()

    out = np.empty_like(x2)
    out[0] = x_hat

    # Derivative smoothing alpha (fixed cutoff)
    a_d = float(_alpha(np.array([cfg.d_cutoff], dtype=np.float32), dt)[0])

    min_c = float(cfg.min_cutoff)
    beta = float(cfg.beta)

    for t in range(1, T):
        # Raw derivative (velocity)
        dx = (x2[t] - x2[t - 1]) / dt

        # Smooth derivative
        dx_hat = a_d * dx + (1.0 - a_d) * dx_hat

        # Speed-adaptive cutoff: faster motion => higher cutoff => less smoothing
        cutoff = min_c + beta * np.abs(dx_hat)

        # Convert cutoff to alpha (per-dimension)
        a = _alpha(cutoff, dt)

        # Smooth signal
        x_hat = a * x2[t] + (1.0 - a) * x_hat
        out[t] = x_hat

    return out.reshape(T, x.shape[1], x.shape[2]).astype(np.float32)


# ============================================================
# 4) Normalization (robust pelvis/torso/rotation)
# ============================================================

def normalize_xy(
    xy: np.ndarray,
    conf: Optional[np.ndarray] = None,
    *,
    conf_gate_for_norm: float = 0.10,
    rotate_shoulders: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalize skeleton per frame:
    - pelvis centering
    - torso-length scaling
    - optional shoulder rotation

    Robustness rules (IMPORTANT):
    - We compute pelvis and shoulder_mid using conf-aware fallbacks.
    - If a keypoint is unreliable, we reuse the last valid estimate.
      This prevents "all joints jump" when hips/shoulders are missing.

    Inputs:
      xy:   [T,J,2]
      conf: [T,J] or None

    conf_gate_for_norm:
      A *lower* threshold than your mask gate is often better here.
      We want stable pelvis/torso when confidence is decent, even if not perfect.
    """
    x = np.asarray(xy, dtype=np.float32).copy()
    T, J, _ = x.shape

    if J <= max(R_HIP, R_SHO):
        return x.astype(np.float32)

    conf_arr: Optional[np.ndarray]
    if conf is None:
        conf_arr = None
    else:
        conf_arr = np.asarray(conf, dtype=np.float32)
        if conf_arr.shape[:2] != (T, J):
            conf_arr = None

    gate = float(conf_gate_for_norm)

    # Helper to decide if a joint is "reliable enough for normalization"
    def ok(t: int, j: int) -> bool:
        if conf_arr is None:
            return np.isfinite(x[t, j, 0]) and np.isfinite(x[t, j, 1])
        return (float(conf_arr[t, j]) >= gate) and np.isfinite(x[t, j, 0]) and np.isfinite(x[t, j, 1])

    # Carry-forward state
    pelvis_prev = np.zeros((2,), dtype=np.float32)
    sh_prev = np.array([0.0, 1.0], dtype=np.float32)  # arbitrary non-zero to avoid 0 torso length
    torso_len_prev = np.float32(1.0)
    ang_prev = np.float32(0.0)

    # Output array
    out = np.empty_like(x, dtype=np.float32)

    for t in range(T):
        # --- pelvis ---
        lh_ok = ok(t, L_HIP)
        rh_ok = ok(t, R_HIP)

        if lh_ok and rh_ok:
            pelvis = 0.5 * (x[t, L_HIP, :] + x[t, R_HIP, :])
        elif lh_ok:
            pelvis = x[t, L_HIP, :]
        elif rh_ok:
            pelvis = x[t, R_HIP, :]
        else:
            pelvis = pelvis_prev  # carry forward if both hips unreliable

        pelvis_prev = pelvis.astype(np.float32)

        # --- shoulder midpoint (for torso vector) ---
        ls_ok = ok(t, L_SHO)
        rs_ok = ok(t, R_SHO)

        if ls_ok and rs_ok:
            sh_mid = 0.5 * (x[t, L_SHO, :] + x[t, R_SHO, :])
        elif ls_ok:
            sh_mid = x[t, L_SHO, :]
        elif rs_ok:
            sh_mid = x[t, R_SHO, :]
        else:
            sh_mid = sh_prev  # carry forward

        sh_prev = sh_mid.astype(np.float32)

        # --- torso length ---
        torso = sh_mid - pelvis
        torso_len = float(np.sqrt(np.sum(torso * torso)))

        if not np.isfinite(torso_len) or torso_len < float(eps):
            torso_len = float(torso_len_prev)

        torso_len = max(float(eps), torso_len)
        torso_len_prev = np.float32(torso_len)

        # --- center + scale all joints ---
        xt = x[t] - pelvis[None, :]
        xt = xt / torso_len

        # --- optional rotate shoulders to horizontal ---
        if rotate_shoulders:
            # Only update angle when both shoulders are reliable (more stable).
            if ls_ok and rs_ok:
                v = xt[R_SHO, :] - xt[L_SHO, :]
                ang = -float(np.arctan2(v[1], v[0]))  # negative => rotate to horizontal
                if np.isfinite(ang):
                    ang_prev = np.float32(ang)

            ca = float(np.cos(float(ang_prev)))
            sa = float(np.sin(float(ang_prev)))

            # 2x2 rotation matrix
            R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
            xt = (xt @ R.T).astype(np.float32)

        out[t] = xt.astype(np.float32)

    return out.astype(np.float32)


# ============================================================
# 5) Mask generation
# ============================================================

def make_mask(xy: np.ndarray, conf: Optional[np.ndarray], conf_gate: float) -> np.ndarray:
    """
    Build binary validity mask [T,J] as uint8.

    Rule:
      valid = (xy finite) AND (conf >= conf_gate)

    Why uint8:
      - NPZ compresses uint8 well
      - easier to multiply features later: mask.astype(float)
    """
    finite = _finite_xy_mask(xy)
    if conf is None:
        return finite.astype(np.uint8)
    return (finite & (np.asarray(conf, dtype=np.float32) >= float(conf_gate))).astype(np.uint8)


# ============================================================
# 6) Main public preprocessing entrypoints
# ============================================================

def preprocess_arrays(
    xy: np.ndarray,
    conf: np.ndarray,
    fps_src: float,
    cfg: PreprocessCfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Preprocess arrays and return:
      joints, conf_resampled, mask, stats

    stats:
      - fps_src, fps, valid_ratio, T_src, T
    """
    # 1) unify missing semantics
    xy0, conf0 = standardize_missing(xy, conf)

    # 2) resample to target fps (deploy timebase)
    xy_rs, conf_rs = resample_to_fps(
        xy0,
        conf0,
        fps_src=float(fps_src),
        fps_tgt=float(cfg.target_fps),
    )

    # 3) smooth (reduce jitter) before computing velocity features
    if bool(cfg.one_euro):
        xy_sm = one_euro_filter_xy(xy_rs, fps=float(cfg.target_fps), cfg=cfg.one_euro_cfg)
    else:
        xy_sm = xy_rs.astype(np.float32, copy=False)

    # 4) normalize (robust pelvis/torso/rotation)
    if bool(cfg.normalize):
        joints = normalize_xy(
            xy_sm,
            conf_rs,
            conf_gate_for_norm=min(0.10, float(cfg.conf_gate)),  # normalization can use a slightly lower gate
            rotate_shoulders=bool(cfg.rotate_shoulders),
        )
    else:
        joints = xy_sm.astype(np.float32, copy=False)

    # 5) mask (final validity)
    mask = make_mask(joints, conf_rs, conf_gate=float(cfg.conf_gate))

    # ratio of valid joints across all frames and joints
    valid_ratio = float(np.mean(mask.astype(np.float32))) if mask.size else 0.0

    stats = {
        "fps_src": float(fps_src),
        "fps": float(cfg.target_fps),
        "valid_ratio": float(valid_ratio),
        "T_src": int(np.asarray(xy).shape[0]),
        "T": int(joints.shape[0]),
    }
    return joints.astype(np.float32), conf_rs.astype(np.float32), mask.astype(np.uint8), stats


def preprocess_npz_file(
    in_path: str,
    out_path: str,
    cfg: PreprocessCfg,
    *,
    fps_default: float = 30.0,
    frame_gate: float = 0.20,
) -> Dict[str, Any]:
    """
    Preprocess one pose_npz file and write a proc_npz file.

    Inputs expected in pose_npz:
      xy   : [T,J,2]
      conf : [T,J]
      fps  : scalar (optional)
      y    : scalar (optional)
      clip_id/relpath/src/size : optional metadata

    frame_gate:
      Frame-level validity threshold.
      frame_mask[t] = 1 if mean(mask[t]) >= frame_gate else 0
    """
    # allow_pickle=False is safer and faster.
    with np.load(in_path, allow_pickle=False) as z:
        xy = np.asarray(z["xy"], dtype=np.float32)
        conf = np.asarray(z["conf"], dtype=np.float32)
        fps_src = _safe_scalar(z, "fps", float(fps_default))

        # Optional metadata (keep defaults if missing)
        y = int(np.array(z["y"]).reshape(-1)[0]) if "y" in z.files else -1

        clip_id = _as_str(z["clip_id"]) if "clip_id" in z.files else os.path.splitext(os.path.basename(in_path))[0]
        relpath = _as_str(z["relpath"]) if "relpath" in z.files else ""
        src = _as_str(z["src"]) if "src" in z.files else ""
        size = np.asarray(z["size"], dtype=np.int32) if "size" in z.files else None

    joints, conf_rs, mask, stats = preprocess_arrays(xy, conf, fps_src=float(fps_src), cfg=cfg)

    # frame_mask is helpful when you want to quickly skip very-bad frames
    # without inspecting every joint.
    per_frame_valid = np.mean(mask.astype(np.float32), axis=1) if mask.size else np.zeros((joints.shape[0],), dtype=np.float32)
    frame_mask = (per_frame_valid >= float(frame_gate)).astype(np.uint8)

    meta = {
        "schema": "proc_npz_v1",
        "cfg": cfg.to_dict(),
        "stats": stats,
        "frame_gate": float(frame_gate),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Atomic write: write tmp then rename
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        joints=joints,
        xy=joints,  # backward compatible alias
        conf=conf_rs,
        mask=mask,
        fps=np.float32(cfg.target_fps),
        fps_src=np.float32(fps_src),
        y=np.int8(y),
        clip_id=str(clip_id),
        relpath=str(relpath),
        src=str(src),
        **({"size": size} if size is not None else {}),
        frame_mask=frame_mask,
        valid_ratio=per_frame_valid.astype(np.float32),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    os.replace(tmp, out_path)

    return {
        "clip_id": clip_id,
        "T": int(joints.shape[0]),
        "fps": float(cfg.target_fps),
        "valid_ratio": float(stats["valid_ratio"]),
    }
