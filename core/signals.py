#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/signals.py

Compute per-window *auxiliary signals* used for post-processing, alerting, and analysis:

  1) quality  : fraction of valid joints in the window (0..1)
  2) lying    : "torso horizontalness" proxy (0..1)
  3) motion   : mean motion magnitude (units depend on motion scaling)

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
  Let v = shoulder_mid - hip_mid in the model coordinate system.
  We compute:
    lying_frame = |dx| / (|dx| + |dy| + eps)
  lying is the mean of lying_frame across valid frames.
  Interpretation:
    - If torso is horizontal: dy small, dx large => lying ~ 1
    - If torso is vertical:   dy large => lying ~ 0

motion:
  mean of motion magnitude over valid joints:
    mag = sqrt(dx^2 + dy^2)
  motion is the mean mag for mask==True entries.

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

    # -------------------------
    # 2) lying: torso horizontalness
    # -------------------------
    lying = 0.0

    # XY channels are always in X[...,0:2] by our feature packing rule.
    xy = X[..., 0:2]  # [T,V,2], already masked (invalid joints are zeroed)

    T, V, _ = xy.shape
    if V > R_HIP and m is not None:
        # Only use frames where BOTH shoulders and BOTH hips are valid.
        # This avoids "lying" being computed on zeros when joints are missing.
        ok = (m[:, L_SHO] & m[:, R_SHO] & m[:, L_HIP] & m[:, R_HIP])  # [T]

        if np.any(ok):
            # Shoulder and hip midpoints per frame
            sh = 0.5 * (xy[:, L_SHO, :] + xy[:, R_SHO, :])   # [T,2]
            hip = 0.5 * (xy[:, L_HIP, :] + xy[:, R_HIP, :])  # [T,2]

            # Torso vector
            v = sh - hip

            # Use absolute x/y to measure direction without sign
            dx = np.abs(v[ok, 0])
            dy = np.abs(v[ok, 1])

            # Horizontalness ratio in [0,1]
            # If dy dominates => close to 0 (upright)
            # If dx dominates => close to 1 (horizontal)
            lying = _safe_mean(dx / (dx + dy + 1e-6))

    # -------------------------
    # 3) motion: mean magnitude over valid joints
    # -------------------------
    motion = 0.0

    # Motion channels are X[...,2:4] only if:
    #   feat_cfg.use_motion is True
    # and our feature tensor has at least 4 channels.
    if feat_cfg.use_motion and X.shape[-1] >= 4 and m is not None:
        mv = X[..., 2:4]                       # [T,V,2] motion (masked)
        mag = np.sqrt(np.sum(mv * mv, axis=-1)) # [T,V] magnitude

        # Only average over valid joints (mask==True)
        motion = _safe_mean(mag[m])

    return WindowSignals(quality=float(quality), lying=float(lying), motion=float(motion))
