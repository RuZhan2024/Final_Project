#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/augment.py

Training-time augmentations for skeleton windows (NumPy-only).

Why augment skeletons?
----------------------
You want the model to handle real-world MediaPipe failure modes:
- left/right flips (camera mirroring)
- pose jitter (noisy keypoints)
- missing joints (occlusion / low confidence)
- missing frames (tracking dropouts)
- contiguous occlusion bursts (someone blocks camera / big motion blur)
- small time shifts (slight timing jitter in streaming)

IMPORTANT pipeline rule
-----------------------
These augmentations are applied to WINDOW arrays *before feature building*.
If joints are changed, motion should be recomputed.

In your pipeline that means:
- Call apply_augmentations(...), then pass motion=None into core/features
  so core/features recomputes motion consistently from the augmented joints.

Input arrays (window-level)
---------------------------
joints: [T, V, 2] float32   (x,y per joint)
conf:   [T, V]    float32 or None
mask:   [T, V]    bool/uint8 or None

Coordinate assumptions
----------------------
Your preprocess step typically outputs pelvis-centered / torso-scaled joints,
so x≈0 at pelvis and values can be negative/positive.

Therefore horizontal flip is implemented as:
  x -> -x
NOT as (1-x). (That would be for image-normalized coords in [0,1].)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# MediaPipe Pose has 33 landmarks by default.
# Left/right swap pairs (indices) so hflip remains anatomically correct.
LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28),
    (29, 30), (31, 32),
]


@dataclass(frozen=True)
class AugCfg:
    """
    Augmentation configuration.

    hflip_p:
      Probability of horizontal flip.

    jitter_std:
      Standard deviation of Gaussian noise added to joints (same unit as joints).
      If your joints are torso-normalized, typical small values: 0.002 ~ 0.02

    jitter_conf_scaled:
      If True and conf exists, noise is scaled by (1 - conf),
      meaning:
        - high confidence points get less noise
        - low confidence points get more noise
      This mimics "uncertain keypoints jitter more".

    mask_joint_p:
      Probability to drop each joint entirely (for all frames).
      Example: 0.05 means each joint has 5% chance of being masked off.

    mask_frame_p:
      Probability to drop each frame entirely (all joints invalid).
      Mimics a frame where pose tracking fails completely.

    occ_p / occ_min_len / occ_max_len:
      Occlusion burst: with probability occ_p, pick a contiguous segment of frames
      with length in [min_len, max_len] and mask all joints in that segment.

    time_shift:
      Random integer shift in [-time_shift, +time_shift].
      Positive shift means "delay": pad with first frame.
      Negative shift means "advance": pad with last frame.

    conf_gate:
      Used only when mask is missing and we must derive a mask from (joints, conf).
    """

    # geometric
    hflip_p: float = 0.0

    # noise
    jitter_std: float = 0.0
    jitter_conf_scaled: bool = True

    # missingness (mask manipulation)
    mask_joint_p: float = 0.0
    mask_frame_p: float = 0.0

    # occlusion burst (contiguous frame dropout)
    occ_p: float = 0.0
    occ_min_len: int = 3
    occ_max_len: int = 10

    # time shift (small temporal jitter)
    time_shift: int = 0

    # mask derivation fallback
    conf_gate: float = 0.20


# ============================================================
# 1) Basic utilities: derive mask / left-right swap / hflip
# ============================================================
def _derive_mask(joints: np.ndarray, conf: Optional[np.ndarray], conf_gate: float) -> np.ndarray:
    """
    Derive validity mask from joints/conf if mask is not provided.

    Rule:
      valid = finite(xy) AND (conf >= conf_gate)

    If conf is None or conf_gate <= 0:
      valid = finite(xy)
    """
    finite = np.isfinite(joints[..., 0]) & np.isfinite(joints[..., 1])
    if conf is None or conf_gate <= 0:
        return finite
    return finite & (conf >= float(conf_gate))


def _swap_lr(
    joints: np.ndarray,
    conf: Optional[np.ndarray],
    mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Swap left/right keypoints in-place using LR_PAIRS.

    This makes horizontal flip anatomically consistent:
    - left shoulder becomes right shoulder, etc.
    """
    for a, b in LR_PAIRS:
        joints[:, [a, b], :] = joints[:, [b, a], :]
        if conf is not None:
            conf[:, [a, b]] = conf[:, [b, a]]
        if mask is not None:
            mask[:, [a, b]] = mask[:, [b, a]]
    return joints, conf, mask


def _hflip(
    joints: np.ndarray,
    conf: Optional[np.ndarray],
    mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Horizontal flip in pelvis-centered coordinates:
      x -> -x

    Then swap left/right joints.
    """
    joints[..., 0] *= -1.0
    return _swap_lr(joints, conf, mask)


# ============================================================
# 2) Time shift
# ============================================================
def _apply_time_shift(
    joints: np.ndarray,
    conf: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    rng: np.random.Generator,
    max_shift: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Shift the whole sequence by a small random offset.

    If s > 0:
      pad with first frame, shift content right
    If s < 0:
      pad with last frame, shift content left

    This simulates small timing misalignment between actions and windows.
    """
    max_shift = int(max_shift)
    if max_shift <= 0:
        return joints, conf, mask

    T = joints.shape[0]
    if T <= 1:
        return joints, conf, mask

    s = int(rng.integers(-max_shift, max_shift + 1))
    if s == 0:
        return joints, conf, mask

    def _shift_arr(a: np.ndarray) -> np.ndarray:
        if s > 0:
            pad = np.repeat(a[:1], s, axis=0)
            return np.concatenate([pad, a[:-s]], axis=0)
        else:
            s2 = -s
            pad = np.repeat(a[-1:], s2, axis=0)
            return np.concatenate([a[s2:], pad], axis=0)

    joints = _shift_arr(joints)
    if conf is not None:
        conf = _shift_arr(conf)
    if mask is not None:
        mask = _shift_arr(mask)

    return joints, conf, mask


# ============================================================
# 3) Mask manipulation: random joint/frame dropout + occlusion bursts
# ============================================================
def _mask_joint_frame(
    mask: np.ndarray,
    rng: np.random.Generator,
    mask_joint_p: float,
    mask_frame_p: float,
) -> np.ndarray:
    """
    Randomly drop:
    - entire joints (all frames)
    - entire frames (all joints)

    We enforce that mask is not all-false (at least one joint at one frame remains true),
    because an all-false mask can make training unstable (all features become zeros).
    """
    m = mask.astype(bool, copy=True)
    T, V = m.shape

    # Drop joints
    if mask_joint_p > 0:
        drop_j = rng.random(V) < float(mask_joint_p)
        if drop_j.any():
            m[:, drop_j] = False

    # Drop frames
    if mask_frame_p > 0:
        drop_t = rng.random(T) < float(mask_frame_p)
        if drop_t.any():
            m[drop_t, :] = False

    # Ensure not completely empty
    if not m.any():
        m[int(rng.integers(0, T)), int(rng.integers(0, V))] = True

    return m


def _occlusion_burst(
    mask: np.ndarray,
    rng: np.random.Generator,
    p: float,
    min_len: int,
    max_len: int,
) -> np.ndarray:
    """
    With probability p, pick a contiguous segment of frames and drop all joints.

    This simulates a real occlusion burst: e.g. person passes behind object,
    tracking drops for several frames in a row.
    """
    if p <= 0 or rng.random() > float(p):
        return mask

    T, V = mask.shape
    if T <= 1:
        return mask

    L = int(rng.integers(int(min_len), int(max_len) + 1))
    L = max(1, min(T, L))

    s = int(rng.integers(0, max(1, T - L + 1)))

    m = mask.astype(bool, copy=True)
    m[s : s + L, :] = False

    # Ensure not completely empty
    if not m.any():
        m[int(rng.integers(0, T)), int(rng.integers(0, V))] = True

    return m


# ============================================================
# 4) Jitter noise
# ============================================================
def _jitter(
    joints: np.ndarray,
    rng: np.random.Generator,
    conf: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    std: float,
    conf_scaled: bool,
) -> np.ndarray:
    """
    Add Gaussian noise to joints.

    If conf_scaled and conf is provided:
      noise *= (1 - conf)
    meaning low-confidence points receive larger noise.

    If mask is provided:
      noise is zero where mask is False
      (so we don't add noise to joints we want to treat as missing)
    """
    std = float(std)
    if std <= 0:
        return joints

    noise = rng.normal(0.0, std, size=joints.shape).astype(joints.dtype)

    if conf_scaled and conf is not None:
        scale = (1.0 - np.clip(conf, 0.0, 1.0)).astype(joints.dtype)[..., None]
        noise *= scale

    if mask is not None:
        noise *= mask.astype(joints.dtype)[..., None]

    return joints + noise


# ============================================================
# 5) Public entrypoint
# ============================================================
def apply_augmentations(
    joints: np.ndarray,
    conf: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    fps: float,
    feat_conf_gate: float,
    *,
    rng: np.random.Generator,
    cfg: AugCfg,
    training: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply training-time augmentations (best-effort).

    Parameters
    ----------
    joints:
      [T,V,2] float32. We always copy because we must not mutate caller arrays.

    conf:
      [T,V] float32 or None.

    mask:
      [T,V] bool/uint8 or None.
      If None, we derive mask from joints/conf using feat_conf_gate.

    fps:
      Provided for future extensions (currently not used).
      Keeping it here avoids breaking call sites.

    feat_conf_gate:
      This should match your feature pipeline conf_gate.
      If mask is missing, we use this to derive mask.

    rng:
      numpy random generator injected by the DataLoader worker for reproducibility.

    cfg:
      AugCfg configuration

    training:
      If False, return inputs unchanged (no copies, fast).
    """
    if not training:
        return joints, conf, mask

    # ---- shape checks (fail fast = easier debugging) ----
    if joints.ndim != 3 or joints.shape[-1] != 2:
        raise ValueError(f"apply_augmentations: joints must be [T,V,2], got {joints.shape}")

    T, V, _ = joints.shape

    if conf is not None and (conf.ndim != 2 or conf.shape != (T, V)):
        raise ValueError(f"apply_augmentations: conf must be [T,V]={T,V}, got {conf.shape}")

    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]
        if m.ndim != 2 or m.shape != (T, V):
            raise ValueError(f"apply_augmentations: mask must be [T,V]={T,V}, got {m.shape}")

    # ---- copy inputs (never mutate caller arrays) ----
    joints = np.array(joints, dtype=np.float32, copy=True)
    conf = (np.array(conf, dtype=np.float32, copy=True) if conf is not None else None)

    # ---- ensure we have a boolean mask for augmentation logic ----
    if mask is None:
        # derive from joints/conf: needed for jitter & occlusion logic
        gate = float(feat_conf_gate) if feat_conf_gate is not None else float(cfg.conf_gate)
        mask = _derive_mask(joints, conf, conf_gate=gate)
    else:
        mask = np.array(mask, dtype=bool, copy=True)

    # 1) Time shift first: it preserves temporal correlations for later ops
    joints, conf, mask = _apply_time_shift(joints, conf, mask, rng, int(cfg.time_shift))

    # 2) Geometric: horizontal flip with probability hflip_p
    if float(cfg.hflip_p) > 0 and rng.random() < float(cfg.hflip_p):
        joints, conf, mask = _hflip(joints, conf, mask)

    # 3) Missingness: independent joint/frame dropout
    if float(cfg.mask_joint_p) > 0 or float(cfg.mask_frame_p) > 0:
        mask = _mask_joint_frame(mask, rng, float(cfg.mask_joint_p), float(cfg.mask_frame_p))

    # 4) Occlusion burst: contiguous segment masked off
    if float(cfg.occ_p) > 0:
        mask = _occlusion_burst(mask, rng, float(cfg.occ_p), int(cfg.occ_min_len), int(cfg.occ_max_len))

    # 5) Noise: add jitter (scaled by conf and zeroed by mask)
    joints = _jitter(joints, rng, conf, mask, float(cfg.jitter_std), bool(cfg.jitter_conf_scaled))

    # ---- numeric safety: never return NaN/inf ----
    joints = np.nan_to_num(joints, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if conf is not None:
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # NOTE:
    # We intentionally do NOT overwrite joints/conf where mask is False.
    # Why?
    # - core/features multiplies by mask, so invalid joints contribute 0 anyway.
    # - Keeping joints untouched can help debugging / visualization.
    #
    # If you want strict “masked joints are zeros”, you can uncomment:
    # joints *= mask.astype(np.float32)[..., None]
    # if conf is not None:
    #     conf *= mask.astype(np.float32)

    return joints, conf, mask
