#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/signals.py

Compute per-window *auxiliary signals* used for post-processing, alerting, and analysis:

  1) quality  : fraction of valid joints in the window (0..1)
  2) lying    : "torso horizontalness" proxy (0..1)
  3) motion   : robust post-motion magnitude (units depend on motion scaling)

Why this file matters
---------------------
These signals must be computed from the SAME representation the model sees.
If the model consumes:
  - pelvis-centered joints
  - masked invalid joints
  - motion scaled by fps (velocity per second)
then signals must use those same rules; otherwise your confirmation logic
or OP fitting can drift away from model behavior.

Implementation detail
---------------------
We call core.features.build_gcn_input(...) because that function is the
"single source of truth" for:
- centering (pelvis)
- mask derivation / usage
- motion recomputation and motion scaling by fps
- confidence channel behavior

So signals remain consistent even when feature flags change.

Signals definitions
-------------------
quality:
  mean(mask) across all frames and joints:
    quality = (#valid joints) / (T * V)

lying (torso horizontalness):
  We want a signal that is HIGH when the person is likely horizontal.

  v = shoulder_mid - hip_mid (in the model coordinate system).
  A more stable horizontalness measure is:
    lying_angle = |dx| / (sqrt(dx^2 + dy^2) + eps)
  which is sin(theta) where theta is the torso angle away from vertical.

  We also compute a simple "bbox aspect" proxy:
    bbox_aspect = (x_range / (y_range + eps))
  and map it into [0,1]. This helps when shoulder/hip angle is noisy.

  Final lying = max(lying_angle_mean, lying_bbox_mean).
  Interpretation:
    - If torso is horizontal: dy small, dx large => lying ~ 1
    - If torso is vertical:   dy large => lying ~ 0

motion:
  Confirmation wants "still after the fall", but the window also contains
  the fall itself (high motion). Using a full-window mean makes confirm
  overly strict and collapses recall.

  We therefore compute motion on the *tail* of the window (last ~1/3), and
  use a robust statistic per-frame:
    frame_motion = median_joints( sqrt(dx^2 + dy^2) )
  motion = mean(frame_motion over tail frames)

Notes:
- If you enable motion_scale_by_fps in features, motion is "per second" velocity.
- If motion is disabled, motion signal returns 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.features import FeatCfg, build_gcn_input


# MediaPipe Pose indices used by this repo.
# We only compute lying if we have these joints.
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24


# ------------------------------------------------------------
# Tail-of-window policy for confirm signals
# ------------------------------------------------------------
# We compute lying/motion on the LAST part of the window so the signal
# reflects the *post-fall* state (lying + still), not the fall dynamics.
_TAIL_FRAC = 0.33
_TAIL_MIN_FRAMES = 8


def _tail_slice(T: int) -> slice:
    """Return a slice selecting the tail frames used for confirm signals."""
    if T <= 0:
        return slice(0, 0)
    k = int(np.ceil(float(T) * float(_TAIL_FRAC)))
    k = max(int(_TAIL_MIN_FRAMES), k)
    k = min(int(T), k)
    return slice(int(T - k), int(T))


@dataclass(frozen=True)
class WindowSignals:
    """
    Container returned by compute_window_signals().

    quality:
      fraction of valid joints in the whole window (0..1)

    lying:
      torso horizontalness proxy (0..1), higher => more horizontal / "lying"

    motion:
      mean motion magnitude over valid joints
    """
    quality: float
    lying: float
    motion: float


def _safe_mean(x: np.ndarray) -> float:
    """
    Mean that is safe under:
    - empty arrays
    - NaN/inf

    Returns 0.0 if no finite values exist.
    """
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return 0.0
    m = np.isfinite(a)
    if not np.any(m):
        return 0.0
    return float(np.mean(a[m]))


def compute_window_signals(
    joints_xy: np.ndarray,
    motion_xy: Optional[np.ndarray],
    conf: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    fps: float,
    feat_cfg: FeatCfg,
) -> WindowSignals:
    """
    Compute (quality, lying, motion) for one window.

    Parameters
    ----------
    joints_xy:
      [T, V, 2] float32
      - window joint positions

    motion_xy:
      [T, V, 2] float32 OR None
      - if None, build_gcn_input will recompute motion consistently

    conf:
      [T, V] float32 OR None
      - optional confidence values

    mask:
      [T, V] bool/uint8 OR None
      - optional validity mask
      - if None, build_gcn_input derives it from xy/conf using feat_cfg.conf_gate

    fps:
      fps for this window (used only when motion needs scaling/recompute)

    feat_cfg:
      feature config controlling centering/motion/conf channels

    Returns
    -------
    WindowSignals(quality, lying, motion)

    Important:
      We intentionally compute signals from build_gcn_input output so that
      signals reflect the exact same representation the model sees.
    """
    # ---- shape checks (fail fast = easier debugging) ----
    if joints_xy.ndim != 3 or joints_xy.shape[-1] != 2:
        raise ValueError(f"compute_window_signals: joints_xy must be [T,V,2], got {joints_xy.shape}")

    # build_gcn_input returns:
    #   X: [T, V, F]  (xy, motion, conf depending on flags)
    #   m: [T, V]     (bool validity mask actually used)
    X, m = build_gcn_input(joints_xy, motion_xy, conf, mask, float(fps), feat_cfg)

    # -------------------------
    # 1) quality: valid fraction
    # -------------------------
    # m is boolean [T,V], True means "valid joint"
    quality = float(np.mean(m.astype(np.float32))) if m is not None else 0.0

    # Tail frames for confirm-related signals
    tail = _tail_slice(int(X.shape[0]))

    # -------------------------
    # 2) lying: torso horizontalness (tail-based)
    # -------------------------
    lying = 0.0

    # XY channels are always in X[...,0:2] by our feature packing rule.
    xy = X[..., 0:2]  # [T,V,2], already masked (invalid joints are zeroed)

    T, V, _ = xy.shape
    if V > R_HIP and m is not None:
        # Only use frames where BOTH shoulders and BOTH hips are valid.
        # This avoids "lying" being computed on zeros when joints are missing.
        ok = (m[:, L_SHO] & m[:, R_SHO] & m[:, L_HIP] & m[:, R_HIP])  # [T]
        ok_tail = ok.copy()
        ok_tail[: tail.start] = False

        lying_angle = 0.0
        if np.any(ok_tail):
            # Shoulder and hip midpoints per frame
            sh = 0.5 * (xy[:, L_SHO, :] + xy[:, R_SHO, :])   # [T,2]
            hip = 0.5 * (xy[:, L_HIP, :] + xy[:, R_HIP, :])  # [T,2]

            # Torso vector
            v = sh - hip

            # Use absolute x/y to measure direction without sign
            dx = np.abs(v[ok_tail, 0])
            dy = np.abs(v[ok_tail, 1])

            # More stable "horizontalness" in [0,1]
            # - upright:  dx small, dy large => ~0
            # - horizontal: dx large => ~1
            lying_angle = _safe_mean(dx / (np.sqrt(dx * dx + dy * dy) + 1e-6))

        # Bounding-box aspect proxy (tail-based), works even if shoulders/hips are noisy.
        # Compute width/height over valid joints for each tail frame.
        lying_bbox = 0.0
        try:
            xy_tail = xy[tail]  # [k,V,2]
            m_tail = m[tail]    # [k,V]
            if xy_tail.size > 0 and np.any(m_tail):
                # Mask invalid joints as NaN so nanmin/nanmax ignore them.
                xt = xy_tail[..., 0].astype(np.float32)
                yt = xy_tail[..., 1].astype(np.float32)
                xt = np.where(m_tail, xt, np.nan)
                yt = np.where(m_tail, yt, np.nan)
                x_rng = np.nanmax(xt, axis=1) - np.nanmin(xt, axis=1)  # [k]
                y_rng = np.nanmax(yt, axis=1) - np.nanmin(yt, axis=1)  # [k]
                aspect = x_rng / (y_rng + 1e-6)
                # Map aspect into [0,1] with a gentle ramp.
                # Typical standing: aspect <~ 0.8
                # Typical lying:    aspect >~ 1.4
                lying_bbox = _safe_mean(np.clip((aspect - 0.8) / (1.4 - 0.8), 0.0, 1.0))
        except Exception:
            lying_bbox = 0.0

        lying = float(max(float(lying_angle), float(lying_bbox)))

    # -------------------------
    # 3) motion: tail-based robust motion magnitude
    # -------------------------
    motion = 0.0

    # Motion channels are X[...,2:4] only if:
    #   feat_cfg.use_motion is True
    # and our feature tensor has at least 4 channels.
    if feat_cfg.use_motion and X.shape[-1] >= 4 and m is not None:
        mv = X[..., 2:4]                        # [T,V,2] motion (masked)
        mag = np.sqrt(np.sum(mv * mv, axis=-1)) # [T,V] magnitude

        # Robust per-frame motion: median over valid joints.
        mag2 = mag.astype(np.float32, copy=True)
        mag2[~m] = np.nan
        frame_med = np.nanmedian(mag2, axis=1)  # [T]
        motion = _safe_mean(frame_med[tail])

    return WindowSignals(quality=float(quality), lying=float(lying), motion=float(motion))
