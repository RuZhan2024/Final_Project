#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/preprocess.py

Shared preprocessing for skeleton (2D pose) sequences.

Design goals
------------
- NumPy-only (usable by offline scripts AND server runtime)
- Deterministic (same input -> same output)
- Robust to missing keypoints (hips/shoulders missing should NOT break normalization)
- Single source of truth: offline + deploy should call the same functions here

Pipeline overview (pose_npz -> proc_npz)
----------------------------------------
Input arrays:
  xy   : [T, J, 2] float32
  conf : [T, J]    float32

Steps:
1) Standardize missing semantics
2) Optional gap fill (short gaps only)
3) Resample to target FPS
4) Smooth (One-Euro OR WMA fallback)
5) Normalize (pelvis-center + scale + optional shoulder rotation)
6) Mask generation

Public entrypoints:
- preprocess_arrays(xy, conf, fps_src, cfg)
- preprocess_npz_file(in_path, out_path, cfg, ...)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ============================================================
# MediaPipe landmark indices (stable across MediaPipe Pose)
# ============================================================
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24


# ============================================================
# Config dataclasses
# ============================================================

@dataclass(frozen=True)
class OneEuroCfg:
    """
    One-Euro filter configuration.

    min_cutoff: base cutoff frequency (Hz)
    beta: speed coefficient (less smoothing when moving fast)
    d_cutoff: derivative cutoff (Hz)
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

    target_fps: resample to this FPS
    conf_gate: confidence threshold for final mask

    gap_fill_max: if >0, fill missing gaps up to this many frames (per joint)
    gap_fill_conf: confidence policy for filled points:
      keep | thr | min_neighbors | linear

    normalize: enable pelvis-center + scaling (+ optional rotation)
    norm_mode: torso | shoulder
    pelvis_fallback: carry | zero
    rotate_shoulders: rotate shoulders to horizontal

    one_euro: enable One-Euro smoothing
    wma_k: weighted moving average window size (used if one_euro=False)
    """
    target_fps: float = 30.0
    conf_gate: float = 0.20
    fill_conf_thr: float = 0.20          # gap fill eligibility
    norm_conf_gate: float = 0.10         # pelvis/shoulder gating for normalization
    gap_fill_max: int = 0
    gap_fill_conf: str = "thr"

    normalize: bool = True
    norm_mode: str = "torso"
    pelvis_fallback: str = "carry"
    rotate_shoulders: bool = False

    one_euro: bool = True
    one_euro_cfg: OneEuroCfg = OneEuroCfg()

    wma_k: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Small helpers
# ============================================================

def _as_str(x: Any) -> str:
    """Convert various np scalar/string forms into a clean Python string."""
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
    """Read a scalar float from an NPZ safely; if missing/malformed => default."""
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


def _atomic_save_npz(out_path: str | Path, **arrays) -> None:
    """
    Atomically write a .npz file.

    NumPy appends '.npz' if filename doesn't end with it, so temp MUST end with '.npz'.
    We use '<stem>.tmp.npz'.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)


# ============================================================
# 1) Standardize missing semantics
# ============================================================

def standardize_missing(xy: np.ndarray, conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rules:
    - conf NaN/inf -> 0
    - if conf <= 0 OR xy non-finite -> xy becomes NaN
    - if xy non-finite -> conf becomes 0
    """
    xy2 = np.asarray(xy, dtype=np.float32).copy()
    conf2 = np.asarray(conf, dtype=np.float32).copy()

    conf2 = np.nan_to_num(conf2, nan=0.0, posinf=0.0, neginf=0.0)
    bad_xy = ~_finite_xy_mask(xy2)
    bad_conf = conf2 <= 0.0
    missing = bad_xy | bad_conf

    xy2[missing] = np.nan
    conf2[bad_xy] = 0.0
    return xy2, conf2


# ============================================================
# 2) Optional gap filling
# ============================================================

def linear_fill_small_gaps(
    xy: np.ndarray,
    conf: np.ndarray,
    *,
    conf_thr: float,
    max_gap: int,
    fill_conf: str = "thr",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill short missing gaps (<= max_gap) by linear interpolation for each joint.

    Returns: xy_out, conf_out, filled_mask (T,J) bool
    """
    if max_gap <= 0:
        return xy, conf, np.zeros(conf.shape, dtype=bool)

    fill_conf = str(fill_conf).lower()
    if fill_conf not in ("keep", "thr", "min_neighbors", "linear"):
        raise ValueError("fill_conf must be one of: keep|thr|min_neighbors|linear")

    xy = np.asarray(xy, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)

    T, J, _ = xy.shape
    out_xy = xy.copy()
    out_conf = conf.copy()
    filled = np.zeros((T, J), dtype=bool)

    # valid = (conf >= float(conf_thr)) & _finite_xy_mask(xy)  # (T,J)
    valid = (conf >= float(conf_thr))  # finiteness handled per-coordinate by np.isfinite(s)

    idx = np.arange(T)

    for j in range(J):
        v = valid[:, j]
        if int(v.sum()) < 2:
            continue

        for c in range(2):
            s = out_xy[:, j, c]
            good = v & np.isfinite(s)
            if int(good.sum()) < 2:
                continue

            miss = ~good
            if not bool(miss.any()):
                continue

            run_starts = np.where(miss & ~np.r_[False, miss[:-1]])[0]
            run_ends = np.where(miss & ~np.r_[miss[1:], False])[0]

            for a, b in zip(run_starts, run_ends):
                gap_len = int(b - a + 1)
                if gap_len > int(max_gap):
                    continue

                left = int(a - 1)
                right = int(b + 1)
                if left < 0 or right >= T:
                    continue
                if not good[left] or not good[right]:
                    continue

                s[a:right] = np.interp(idx[a:right], [left, right], [s[left], s[right]])
                filled[a:right, j] = True

            out_xy[:, j, c] = s

        if bool(filled[:, j].any()) and fill_conf != "keep":
            miss_j = filled[:, j]

            if fill_conf == "thr":
                out_conf[miss_j, j] = np.maximum(out_conf[miss_j, j], float(conf_thr)).astype(np.float32)
            else:
                run_starts = np.where(miss_j & ~np.r_[False, miss_j[:-1]])[0]
                run_ends = np.where(miss_j & ~np.r_[miss_j[1:], False])[0]

                for a, b in zip(run_starts, run_ends):
                    left = int(a - 1)
                    right = int(b + 1)
                    if left < 0 or right >= T:
                        continue

                    cl = float(out_conf[left, j])
                    cr = float(out_conf[right, j])

                    if fill_conf == "min_neighbors":
                        val = max(float(min(cl, cr)), float(conf_thr))
                        out_conf[a:right, j] = val
                    elif fill_conf == "linear":
                        out_conf[a:right, j] = np.interp(idx[a:right], [left, right], [cl, cr]).astype(np.float32)

    return out_xy.astype(np.float32), out_conf.astype(np.float32), filled


# ============================================================
# 3) Resample to target FPS
# ============================================================

def resample_to_fps(
    xy: np.ndarray,
    conf: np.ndarray,
    fps_src: float,
    fps_tgt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a sequence from fps_src to fps_tgt.

    - xy: linear interpolation when endpoints are finite (per coordinate)
          otherwise keep NaN.
    - conf: nearest-neighbor sampling, but set to 0 where xy is non-finite.
    """
    xy = np.asarray(xy, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)

    T = int(xy.shape[0])
    if T <= 1:
        return xy.copy(), conf.copy()

    fs = float(fps_src) if float(fps_src) > 0 else float(fps_tgt)
    ft = float(fps_tgt) if float(fps_tgt) > 0 else float(fps_src)

    if fs <= 0 or ft <= 0 or abs(fs - ft) < 1e-6:
        return xy.copy(), conf.copy()

    # Preserve duration based on (T-1)/fps
    dur_s = float((T - 1) / fs)
    Tt = int(round(dur_s * ft)) + 1
    Tt = max(1, Tt)

    t_tgt = np.arange(Tt, dtype=np.float32) / ft
    idx_f = t_tgt * fs

    i0 = np.floor(idx_f).astype(np.int32)
    i0 = np.clip(i0, 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)

    a = (idx_f - i0).astype(np.float32)        # (Tt,)
    a3 = a[:, None, None]                      # (Tt,1,1)

    # Gather endpoints WITHOUT nan_to_num
    xy0 = xy[i0]                               # (Tt,J,2)
    xy1 = xy[i1]                               # (Tt,J,2)

    xy_tgt = (1.0 - a3) * xy0 + a3 * xy1       # (Tt,J,2)

    ok_x = np.isfinite(xy0[..., 0]) & np.isfinite(xy1[..., 0])   # (Tt,J)
    ok_y = np.isfinite(xy0[..., 1]) & np.isfinite(xy1[..., 1])   # (Tt,J)

    xy_tgt[..., 0] = np.where(ok_x, xy_tgt[..., 0], np.nan).astype(np.float32)
    xy_tgt[..., 1] = np.where(ok_y, xy_tgt[..., 1], np.nan).astype(np.float32)

    # Nearest-neighbor conf, then zero out where xy is non-finite
    idx_nn = np.clip(np.round(idx_f).astype(np.int32), 0, T - 1)
    conf_tgt = conf[idx_nn].astype(np.float32, copy=True)  # (Tt,J)
    conf_tgt = np.where(np.isfinite(conf_tgt), conf_tgt, 0.0).astype(np.float32)
    conf_tgt[~_finite_xy_mask(xy_tgt)] = 0.0

    return xy_tgt.astype(np.float32), conf_tgt.astype(np.float32)


# ============================================================
# 4) Smoothing
# ============================================================

def _alpha(cutoff_hz: np.ndarray, dt: float) -> np.ndarray:
    cutoff_hz = np.asarray(cutoff_hz, dtype=np.float32)
    cutoff_hz = np.maximum(cutoff_hz, 1e-6)
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


def _ffill_nan_2d(x2: np.ndarray) -> np.ndarray:
    """
    Forward-fill NaNs/inf in a (T,D) float array per dimension.
    - Leading missing values are filled with the first valid value.
    - If a dimension has no valid values at all, fill with 0.
    """
    x2 = np.asarray(x2, dtype=np.float32)
    T, D = x2.shape
    out = x2.copy()

    for d in range(D):
        s = out[:, d]
        valid = np.isfinite(s)
        if not bool(valid.any()):
            out[:, d] = 0.0
            continue

        first = int(np.argmax(valid))
        s[:first] = s[first]

        for t in range(first + 1, T):
            if not np.isfinite(s[t]):
                s[t] = s[t - 1]

        out[:, d] = s

    return out


def one_euro_filter_xy(xy: np.ndarray, fps: float, cfg: OneEuroCfg) -> np.ndarray:
    x = np.asarray(xy, dtype=np.float32)
    T = int(x.shape[0])
    if T <= 2:
        return x.copy()

    fs = float(fps) if float(fps) > 0 else 30.0
    dt = 1.0 / fs

    D = int(x.shape[1] * x.shape[2])
    x2 = x.reshape(T, D)

    missing = ~np.isfinite(x2)
    x2_filled = _ffill_nan_2d(x2)

    dx_hat = np.zeros((D,), dtype=np.float32)
    x_hat = x2_filled[0].copy()

    out = np.empty_like(x2_filled)
    out[0] = x_hat

    a_d = float(_alpha(np.array([cfg.d_cutoff], dtype=np.float32), dt)[0])
    min_c = float(cfg.min_cutoff)
    beta = float(cfg.beta)

    for t in range(1, T):
        miss_t = missing[t]

        dx = (x2_filled[t] - x2_filled[t - 1]) / dt
        dx = np.where(miss_t, 0.0, dx).astype(np.float32)

        dx_hat = a_d * dx + (1.0 - a_d) * dx_hat
        cutoff = min_c + beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)

        x_next = a * x2_filled[t] + (1.0 - a) * x_hat
        x_hat = np.where(miss_t, x_hat, x_next).astype(np.float32)

        out[t] = x_hat

    out[missing] = np.nan
    return out.reshape(T, x.shape[1], x.shape[2]).astype(np.float32)


def smooth_weighted_moving_average(xy: np.ndarray, conf: np.ndarray, *, conf_thr: float, k: int) -> np.ndarray:
    k = int(k)
    if k <= 1:
        return np.asarray(xy, dtype=np.float32)
    if k % 2 == 0:
        k += 1

    x = np.asarray(xy, dtype=np.float32)
    c = np.asarray(conf, dtype=np.float32)

    T, J, C = x.shape
    out = np.full_like(x, np.nan, dtype=np.float32)

    w = np.where((c >= float(conf_thr)) & np.isfinite(c), c, 0.0).astype(np.float32)
    half = k // 2

    x_pad = np.pad(x, ((half, half), (0, 0), (0, 0)), mode="edge")
    w_pad = np.pad(w, ((half, half), (0, 0)), mode="edge")

    for t in range(T):
        w_win = w_pad[t : t + k]
        x_win = x_pad[t : t + k]

        nan_mask = ~np.isfinite(x_win[..., 0]) | ~np.isfinite(x_win[..., 1])
        w_eff = w_win.copy()
        w_eff[nan_mask] = 0.0

        denom = w_eff.sum(axis=0)
        ok = denom > 1e-8
        if not bool(ok.any()):
            continue

        for cc in range(C):
            num = (x_win[..., cc] * w_eff).sum(axis=0)
            out[t, ok, cc] = num[ok] / denom[ok]

    return out.astype(np.float32)


# ============================================================
# 5) Normalization (robust pelvis/scale/rotation)
# ============================================================

def normalize_xy(
    xy: np.ndarray,
    conf: Optional[np.ndarray] = None,
    *,
    conf_gate_for_norm: float = 0.10,
    rotate_shoulders: bool = False,
    scale_mode: str = "torso",
    pelvis_fallback: str = "carry",
    eps: float = 1e-6,
) -> np.ndarray:
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
    scale_mode = str(scale_mode).lower()
    pelvis_fallback = str(pelvis_fallback).lower()

    if scale_mode not in ("torso", "shoulder"):
        scale_mode = "torso"
    if pelvis_fallback not in ("carry", "zero"):
        pelvis_fallback = "carry"

    def ok(t: int, j: int) -> bool:
        if conf_arr is None:
            return np.isfinite(x[t, j, 0]) and np.isfinite(x[t, j, 1])
        return (float(conf_arr[t, j]) >= gate) and np.isfinite(x[t, j, 0]) and np.isfinite(x[t, j, 1])

    pelvis_prev = np.zeros((2,), dtype=np.float32)
    sh_mid_prev = np.array([0.0, 1.0], dtype=np.float32)
    scale_prev = np.float32(1.0)
    ang_prev = np.float32(0.0)

    out = np.empty_like(x, dtype=np.float32)

    for t in range(T):
        lh_ok = ok(t, L_HIP)
        rh_ok = ok(t, R_HIP)

        if lh_ok and rh_ok:
            pelvis = 0.5 * (x[t, L_HIP, :] + x[t, R_HIP, :])
        elif lh_ok:
            pelvis = x[t, L_HIP, :]
        elif rh_ok:
            pelvis = x[t, R_HIP, :]
        else:
            pelvis = pelvis_prev if pelvis_fallback == "carry" else np.zeros((2,), dtype=np.float32)

        pelvis_prev = pelvis.astype(np.float32)

        ls_ok = ok(t, L_SHO)
        rs_ok = ok(t, R_SHO)

        if ls_ok and rs_ok:
            sh_mid = 0.5 * (x[t, L_SHO, :] + x[t, R_SHO, :])
        elif ls_ok:
            sh_mid = x[t, L_SHO, :]
        elif rs_ok:
            sh_mid = x[t, R_SHO, :]
        else:
            sh_mid = sh_mid_prev

        sh_mid_prev = sh_mid.astype(np.float32)

        if scale_mode == "shoulder":
            if ls_ok and rs_ok:
                v = x[t, R_SHO, :] - x[t, L_SHO, :]
                s = float(np.sqrt(np.sum(v * v)))
            else:
                s = float(scale_prev)
        else:
            v = sh_mid - pelvis
            s = float(np.sqrt(np.sum(v * v)))

        if not np.isfinite(s) or s < float(eps):
            s = float(scale_prev)
        s = max(float(eps), s)
        scale_prev = np.float32(s)

        xt = (x[t] - pelvis[None, :]) / s

        if rotate_shoulders:
            if ls_ok and rs_ok:
                v2 = xt[R_SHO, :] - xt[L_SHO, :]
                ang = -float(np.arctan2(v2[1], v2[0]))
                if np.isfinite(ang):
                    ang_prev = np.float32(ang)

            ca = float(np.cos(float(ang_prev)))
            sa = float(np.sin(float(ang_prev)))
            R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
            xt = (xt @ R.T).astype(np.float32)

        out[t] = xt.astype(np.float32)

    return out.astype(np.float32)


# ============================================================
# 6) Mask generation
# ============================================================

def make_mask(xy: np.ndarray, conf: Optional[np.ndarray], conf_gate: float) -> np.ndarray:
    finite = _finite_xy_mask(xy)
    if conf is None:
        return finite.astype(np.uint8)
    return (finite & (np.asarray(conf, dtype=np.float32) >= float(conf_gate))).astype(np.uint8)


# ============================================================
# 7) Public entrypoints
# ============================================================

def preprocess_arrays(
    xy: np.ndarray,
    conf: np.ndarray,
    fps_src: float,
    cfg: PreprocessCfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    xy0, conf0 = standardize_missing(xy, conf)

    filled_points = 0
    if int(cfg.gap_fill_max) > 0:
        xy0, conf0, filled = linear_fill_small_gaps(
            xy0,
            conf0,
            conf_thr=float(cfg.fill_conf_thr),
            max_gap=int(cfg.gap_fill_max),
            fill_conf=str(cfg.gap_fill_conf),
        )
        filled_points = int(filled.sum())

    xy_rs, conf_rs = resample_to_fps(
        xy0,
        conf0,
        fps_src=float(fps_src),
        fps_tgt=float(cfg.target_fps),
    )

    if bool(cfg.one_euro):
        xy_sm = one_euro_filter_xy(xy_rs, fps=float(cfg.target_fps), cfg=cfg.one_euro_cfg)
    elif int(cfg.wma_k) > 1:
        xy_sm = smooth_weighted_moving_average(xy_rs, conf_rs, conf_thr=float(cfg.conf_gate), k=int(cfg.wma_k))
    else:
        xy_sm = xy_rs.astype(np.float32, copy=False)

    if bool(cfg.normalize):
        joints = normalize_xy(
            xy_sm,
            conf_rs,
            conf_gate_for_norm=float(cfg.norm_conf_gate),
            rotate_shoulders=bool(cfg.rotate_shoulders),
            scale_mode=str(cfg.norm_mode),
            pelvis_fallback=str(cfg.pelvis_fallback),
        )
    else:
        joints = xy_sm.astype(np.float32, copy=False)

    mask = make_mask(joints, conf_rs, conf_gate=float(cfg.conf_gate))
    valid_ratio = float(np.mean(mask.astype(np.float32))) if mask.size else 0.0

    stats = {
        "fps_src": float(fps_src),
        "fps": float(cfg.target_fps),
        "valid_ratio": float(valid_ratio),
        "T_src": int(np.asarray(xy).shape[0]),
        "T": int(joints.shape[0]),
        "filled_points": int(filled_points),
    }
    return joints.astype(np.float32), conf_rs.astype(np.float32), mask.astype(np.uint8), stats


def preprocess_npz_file(
    in_path: str,
    out_path: str,
    cfg: PreprocessCfg,
    *,
    fps_default: float = 30.0,
    frame_gate: float = 0.20,
    zero_bad_frames: bool = False,
) -> Dict[str, Any]:
    with np.load(in_path, allow_pickle=False) as z:
        if "xy" not in z.files or "conf" not in z.files:
            raise KeyError(f"{in_path}: missing required keys 'xy'/'conf'")

        xy = np.asarray(z["xy"], dtype=np.float32)
        conf = np.asarray(z["conf"], dtype=np.float32)

        extras = {k: z[k] for k in z.files if k not in ("xy", "conf")}

        fps_src = _safe_scalar(z, "fps", float(fps_default))
        y = int(np.array(z["y"]).reshape(-1)[0]) if "y" in z.files else -1

        clip_id = _as_str(z["clip_id"]) if "clip_id" in z.files else os.path.splitext(os.path.basename(in_path))[0]
        relpath = _as_str(z["relpath"]) if "relpath" in z.files else ""
        src = _as_str(z["src"]) if "src" in z.files else ""
        size = np.asarray(z["size"], dtype=np.int32) if "size" in z.files else None

    joints, conf_rs, mask, stats = preprocess_arrays(xy, conf, fps_src=float(fps_src), cfg=cfg)

    per_frame_valid = (
        np.mean(mask.astype(np.float32), axis=1)
        if mask.size
        else np.zeros((joints.shape[0],), dtype=np.float32)
    )
    frame_mask = (per_frame_valid >= float(frame_gate)).astype(np.uint8)

    if bool(zero_bad_frames):
        bad = frame_mask == 0
        if bool(np.any(bad)):
            joints = joints.copy()
            conf_rs = conf_rs.copy()
            mask = mask.copy()

            joints[bad, :, :] = np.nan
            conf_rs[bad, :] = 0.0
            mask[bad, :] = 0

            per_frame_valid = per_frame_valid.copy()
            per_frame_valid[bad] = 0.0

    meta = {
        "schema": "proc_npz_v1",
        "cfg": cfg.to_dict(),
        "stats": stats,
        "frame_gate": float(frame_gate),
        "zero_bad_frames": bool(zero_bad_frames),
    }

    reserved = {
        "joints", "xy", "conf", "mask",
        "fps", "fps_src", "y",
        "clip_id", "relpath", "src", "size",
        "frame_mask", "valid_ratio",
        "preprocess", "meta",
    }
    extras_keep = {k: v for k, v in extras.items() if k not in reserved}

    _atomic_save_npz(
        out_path,
        joints=joints,
        xy=joints,
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
        preprocess=json.dumps(meta, ensure_ascii=False),
        meta=json.dumps(meta, ensure_ascii=False),
        **extras_keep,
    )

    return {
        "clip_id": clip_id,
        "T": int(joints.shape[0]),
        "fps": float(cfg.target_fps),
        "valid_ratio": float(stats.get("valid_ratio", 0.0)),
        "filled_points": int(stats.get("filled_points", 0)),
    }
