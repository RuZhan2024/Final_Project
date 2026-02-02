#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/features.py

Single source of truth for turning WINDOW NPZ files into model inputs.

Why this file matters
---------------------
Every downstream stage (training, evaluation, mining, replay) must interpret a
window NPZ the same way. If different scripts build slightly different features,
your offline metrics won’t match deployment behavior.

This module standardizes:
- How we read window NPZ schema (new + legacy keys)
- How we derive mask (valid joints) consistently
- How we center skeletons (pelvis-centered) consistently
- How we compute motion (per-second velocity) consistently
- How we pack features into:
    * GCN format  : [T, V, F]
    * TCN format  : [T, C] (flattened joints/features)

Window NPZ schema (preferred)
-----------------------------
- joints : float32 [T, V, 2]   (x,y per joint)
- motion : float32 [T, V, 2]   (dx,dy per joint) OPTIONAL
- conf   : float32 [T, V]      confidence/visibility OPTIONAL
- mask   : uint8/bool [T, V]   joint valid mask OPTIONAL
- fps    : float scalar        OPTIONAL
- y      : int scalar          0/1 labeled, -1 unlabeled OPTIONAL
- video_id / seq_id / seq_stem : string-ish OPTIONAL
- w_start / w_end : int scalar OPTIONAL (end is inclusive in your pipeline)

Legacy supported keys
---------------------
- xy instead of joints
- joints_conf instead of conf

Important conventions
---------------------
- Mask rule (if we derive it):
    valid = finite(xy) AND (conf >= conf_gate)
- If pelvis-centering is enabled:
    we center xy per-frame by pelvis midpoint (joint 23 & 24 for MediaPipe Pose).
- Motion meaning:
    velocity per second:
        motion[t] = (xy[t] - xy[t-1]) * fps
    motion[0] = 0
  If we pelvis-center, we recompute motion AFTER centering so motion is consistent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ============================================================
# 1) Feature configuration
# ============================================================
@dataclass(frozen=True)
class FeatCfg:
    """
    Feature flags shared across training + eval.

    center:
      - "pelvis": subtract pelvis center from each frame (translation invariance)
      - "none"  : keep raw coordinates

    use_motion:
      - True: include dx,dy per joint (velocity feature)
      - False: do not include motion channels (or return zeros)

    use_conf_channel:
      - True: append confidence as an extra feature channel
      - False: ignore confidence in the feature tensor (mask still may use conf)

    motion_scale_by_fps:
      - True: motion is velocity per second (multiply frame-to-frame delta by fps)
      - False: motion is per-frame delta

    conf_gate:
      - confidence threshold used to consider a joint “valid” when deriving mask

    use_precomputed_mask:
      - True: if NPZ provides mask, trust it (fast, consistent with preprocess)
      - False: always derive mask from xy/conf using conf_gate
    """

    center: str = "pelvis"
    use_motion: bool = True
    use_conf_channel: bool = True
    motion_scale_by_fps: bool = True
    conf_gate: float = 0.20
    use_precomputed_mask: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (useful for checkpoints/yaml)."""
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FeatCfg":
        """Restore from dict safely (missing keys fall back to defaults)."""
        if not isinstance(d, dict):
            return FeatCfg()
        return FeatCfg(
            center=str(d.get("center", "pelvis")),
            use_motion=bool(d.get("use_motion", True)),
            use_conf_channel=bool(d.get("use_conf_channel", True)),
            motion_scale_by_fps=bool(d.get("motion_scale_by_fps", True)),
            conf_gate=float(d.get("conf_gate", 0.20)),
            use_precomputed_mask=bool(d.get("use_precomputed_mask", True)),
        )


# ============================================================
# 2) Metadata returned when reading a window NPZ
# ============================================================
@dataclass
class WindowMeta:
    """
    Small container holding window metadata (for traceability & debugging).

    path:
      path to the NPZ file

    video_id:
      stable identifier for grouping windows back to the original video/sequence.
      We try keys in order: video_id -> seq_id -> seq_stem -> stem -> ""

    w_start / w_end:
      start and end frame indices (w_end is inclusive in your repo)

    fps:
      stored fps (processed time base)

    y:
      label: 0/1 for labeled windows, -1 for unlabeled windows
    """

    path: str
    video_id: str
    w_start: int
    w_end: int
    fps: float
    y: int  # -1 for unlabeled

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# 3) Small NPZ parsing helpers
# ============================================================
def _as_str(x: Any) -> str:
    """
    Convert a value from NPZ into a Python string.

    NPZ may store strings as:
      - bytes
      - scalar numpy array
      - 1-element numpy array
    """
    try:
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        if isinstance(x, np.ndarray):
            if x.shape == ():
                return str(x.item())
            if x.size == 1:
                return str(x.reshape(-1)[0].item())
        return str(x)
    except Exception:
        return str(x)


def _safe_scalar(z: np.lib.npyio.NpzFile, key: str, default: float) -> float:
    """Read a float scalar from NPZ, fallback to default."""
    if key not in z.files:
        return float(default)
    try:
        return float(np.array(z[key]).reshape(-1)[0])
    except Exception:
        return float(default)


def _safe_int(z: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    """Read an int scalar from NPZ, fallback to default."""
    if key not in z.files:
        return int(default)
    try:
        return int(np.array(z[key]).reshape(-1)[0])
    except Exception:
        return int(default)


def _safe_array(z: np.lib.npyio.NpzFile, key: str, dtype, *, allow_missing: bool = False) -> Optional[np.ndarray]:
    """
    Read an array from NPZ as dtype.

    If allow_missing=True and key not found -> return None.
    """
    if key not in z.files:
        if allow_missing:
            return None
        raise KeyError(f"Missing key {key!r}")
    return np.array(z[key], dtype=dtype, copy=False)


# ============================================================
# 4) Geometry helpers (pelvis center + mask + motion)
# ============================================================
def _pelvis_center(joints_xy: np.ndarray) -> np.ndarray:
    """
    Compute pelvis center per frame: average of left/right hip.

    MediaPipe Pose indices:
      L_HIP = 23
      R_HIP = 24

    Returns:
      center: [T, 1, 2] so it can broadcast-subtract from [T,V,2].
    """
    V = joints_xy.shape[1]
    if V > 24:
        return 0.5 * (joints_xy[:, 23:24, :] + joints_xy[:, 24:25, :])
    if V > 23:
        return joints_xy[:, 23:24, :]
    # fallback: if not enough joints, use joint 0
    return joints_xy[:, 0:1, :]


def _derive_mask(joints_xy: np.ndarray, conf: Optional[np.ndarray], conf_gate: float) -> np.ndarray:
    """
    Derive joint-valid mask.

    Rule:
      finite(xy) AND (conf >= conf_gate)
    If conf is missing or conf_gate<=0:
      finite(xy) only.

    Output:
      mask bool [T,V]
    """
    finite = np.isfinite(joints_xy[..., 0]) & np.isfinite(joints_xy[..., 1])
    if conf is None or conf_gate <= 0:
        return finite
    return finite & (conf >= float(conf_gate))


def _compute_motion(joints_xy: np.ndarray, fps: float, scale_by_fps: bool) -> np.ndarray:
    """
    Compute motion feature from positions.

    Definition:
      delta[t] = joints[t] - joints[t-1]
      delta[0] = 0
      if scale_by_fps: delta *= fps  (velocity per second)

    Output:
      motion float32 [T,V,2]
    """
    m = np.zeros_like(joints_xy, dtype=np.float32)
    m[1:] = joints_xy[1:] - joints_xy[:-1]
    m[0] = 0.0
    if scale_by_fps and fps > 0:
        m = m * float(fps)
    return m.astype(np.float32, copy=False)


def _mask_shape_ok(mask: np.ndarray, T: int, V: int) -> bool:
    """
    Verify precomputed mask has compatible shape.

    Accept:
      - [T,V]
      - [T,V,1] (we squeeze)
    """
    if mask.ndim == 2 and mask.shape == (T, V):
        return True
    if mask.ndim == 3 and mask.shape[0] == T and mask.shape[1] == V:
        return True
    return False


# ============================================================
# 5) Read one WINDOW NPZ
# ============================================================
def read_window_npz(
    path: str,
    fps_default: float = 30.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float, WindowMeta]:
    """
    Read arrays from a window NPZ.

    Returns:
      joints_xy: float32 [T,V,2]
      motion_xy: float32 [T,V,2] or None
      conf     : float32 [T,V] or None
      mask     : bool [T,V] or None
      fps      : float
      meta     : WindowMeta

    Notes:
    - If keys are missing, we return None for optional arrays.
    - We do NOT apply centering/motion/masking here. This function only reads.
      Feature building happens in build_gcn_input/build_tcn_input.
    """
    with np.load(path, allow_pickle=False) as z:
        # ---- positions ----
        if "joints" in z.files:
            joints = _safe_array(z, "joints", np.float32)
        elif "xy" in z.files:
            joints = _safe_array(z, "xy", np.float32)
        else:
            raise KeyError(f"Missing joints/xy in {path}")

        # Basic shape check helps catch broken window files early.
        if joints.ndim != 3 or joints.shape[-1] != 2:
            raise ValueError(f"{path}: expected joints shape [T,V,2], got {joints.shape}")

        # ---- motion (optional) ----
        motion = _safe_array(z, "motion", np.float32, allow_missing=True)

        # ---- confidence (optional) ----
        conf = None
        if "conf" in z.files:
            conf = _safe_array(z, "conf", np.float32)
        elif "joints_conf" in z.files:
            conf = _safe_array(z, "joints_conf", np.float32)

        # ---- mask (optional) ----
        mask = None
        if "mask" in z.files:
            raw_mask = np.array(z["mask"], copy=False)
            # Some code writes [T,V,1]; squeeze it.
            if raw_mask.ndim == 3:
                raw_mask = raw_mask[..., 0]
            mask = raw_mask.astype(bool, copy=False)

        # ---- fps ----
        fps = _safe_scalar(z, "fps", fps_default)

        # ---- metadata strings ----
        # Try several keys in order (whatever is present in the window NPZ).
        if "video_id" in z.files:
            video_id = _as_str(z["video_id"])
        elif "seq_id" in z.files:
            video_id = _as_str(z["seq_id"])
        elif "seq_stem" in z.files:
            video_id = _as_str(z["seq_stem"])
        elif "stem" in z.files:
            video_id = _as_str(z["stem"])
        else:
            video_id = ""

        # ---- window indices ----
        w_start = _safe_int(z, "w_start", 0)
        # Your pipeline treats w_end as inclusive end index.
        w_end = _safe_int(z, "w_end", w_start + int(joints.shape[0]) - 1)

        # ---- label ----
        y = _safe_int(z, "y", -1)

        meta = WindowMeta(
            path=str(path),
            video_id=str(video_id or ""),
            w_start=int(w_start),
            w_end=int(w_end),
            fps=float(fps),
            y=int(y),
        )

    return joints, motion, conf, mask, float(fps), meta


# ============================================================
# 6) Build GCN input: [T,V,F]
# ============================================================
def build_gcn_input(
    joints_xy: np.ndarray,
    motion_xy: Optional[np.ndarray],
    conf: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    fps: float,
    feat_cfg: FeatCfg,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw arrays into a GCN input tensor.

    Output:
      X: float32 [T, V, F]
      mask_used: bool [T, V]

    Feature packing order (stable):
      X[..., 0:2]   = xy
      X[..., 2:4]   = motion (if use_motion)
      X[..., -1]    = conf (if use_conf_channel)

    Why stable ordering matters:
    - Your checkpoints assume channel order is consistent across runs.
    """
    # 1) Numeric safety: remove NaN/inf so tensor ops never explode.
    joints_xy = np.nan_to_num(joints_xy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if conf is not None:
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    T, V, _ = joints_xy.shape

    # 2) Decide which mask to use:
    #    - if precomputed mask exists and shape matches, use it
    #    - else derive from xy/conf
    if feat_cfg.use_precomputed_mask and mask is not None and _mask_shape_ok(mask, T, V):
        m = mask.astype(bool, copy=False)
    else:
        m = _derive_mask(joints_xy, conf, feat_cfg.conf_gate)

    # 3) Centering (translation invariance)
    if feat_cfg.center == "pelvis":
        joints_xy = joints_xy - _pelvis_center(joints_xy)

    # 4) Motion:
    #    Important: if we pelvis-center, recompute motion AFTER centering
    #    because subtracting a per-frame pelvis center changes frame-to-frame deltas.
    if feat_cfg.use_motion:
        if motion_xy is None or feat_cfg.center == "pelvis":
            motion_xy = _compute_motion(joints_xy, fps, feat_cfg.motion_scale_by_fps)
        else:
            motion_xy = np.nan_to_num(motion_xy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            if feat_cfg.motion_scale_by_fps and fps > 0:
                motion_xy = motion_xy * float(fps)
    else:
        motion_xy = np.zeros_like(joints_xy, dtype=np.float32)

    # 5) Apply mask:
    #    We multiply xy/motion by mask so invalid joints contribute 0.
    #    This also keeps numerical scale stable.
    mm = m.astype(np.float32)[..., None]  # [T,V,1] for broadcast
    joints_xy = joints_xy * mm
    motion_xy = motion_xy * mm

    # If conf exists, also zero it where joint is invalid
    if conf is not None:
        conf = conf * m.astype(np.float32)

    # 6) Pack features
    parts = [joints_xy]  # [T,V,2]

    if feat_cfg.use_motion:
        parts.append(motion_xy)  # [T,V,2]

    if feat_cfg.use_conf_channel:
        if conf is None:
            # If conf is missing, use ones (so model isn't punished for missing conf key)
            conf = np.ones((T, V), dtype=np.float32)
        parts.append(conf[..., None].astype(np.float32))  # [T,V,1]

    X = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
    return X, m


# ============================================================
# 7) Build TCN input: [T,C] (flattened)
# ============================================================
def build_tcn_input(
    joints_xy: np.ndarray,
    motion_xy: Optional[np.ndarray],
    conf: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    fps: float,
    feat_cfg: FeatCfg,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert arrays into a TCN input tensor.

    Output:
      Xt: float32 [T, C]
      mask_used: bool [T, V]

    Flattening rule (stable channel ordering):
      - xy     -> [T, V*2]
      - motion -> [T, V*2]  (if enabled)
      - conf   -> [T, V]    (if enabled)

    Why conf is V (not V*1 duplicated):
    - it’s already one scalar per joint
    """
    Xg, m = build_gcn_input(joints_xy, motion_xy, conf, mask, fps, feat_cfg)
    T, V, F = Xg.shape

    # xy always occupies first 2 channels per joint
    xy_flat = Xg[..., 0:2].reshape(T, V * 2)
    parts = [xy_flat]

    # motion occupies next 2 channels per joint (only if enabled)
    if feat_cfg.use_motion:
        motion_flat = Xg[..., 2:4].reshape(T, V * 2)
        parts.append(motion_flat)

    # conf occupies last channel (only if enabled)
    if feat_cfg.use_conf_channel:
        conf_flat = Xg[..., -1].reshape(T, V)
        parts.append(conf_flat)

    Xt = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    return Xt, m
