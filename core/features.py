#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/features.py

Single source of truth for turning a window NPZ into model inputs.

Window NPZ keys (new schema, preferred):
  - joints : [T,V,2] float32
  - motion : [T,V,2] float32 (optional)
  - conf   : [T,V]   float32 (optional)
  - mask   : [T,V]   uint8/bool (optional)
  - fps    : scalar  (optional)
  - y      : 0/1 (labeled) or -1 (unlabeled)

We support older keys:
  - xy instead of joints
  - joints_conf instead of conf
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FeatCfg:
    center: str = "pelvis"           # "pelvis" | "none"
    use_motion: bool = True
    use_conf_channel: bool = True
    motion_scale_by_fps: bool = True
    conf_gate: float = 0.20
    use_precomputed_mask: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FeatCfg":
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


@dataclass
class WindowMeta:
    path: str
    video_id: str
    w_start: int
    w_end: int
    fps: float
    y: int  # -1 for unlabeled

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_str(x: Any) -> str:
    try:
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        if isinstance(x, np.ndarray) and x.shape == ():
            return str(x.item())
        if isinstance(x, np.ndarray) and x.size == 1:
            return str(x.reshape(-1)[0].item())
        return str(x)
    except Exception:
        return str(x)


def _safe_scalar(z, key: str, default: float) -> float:
    if key not in z.files:
        return float(default)
    try:
        return float(np.array(z[key]).reshape(-1)[0])
    except Exception:
        return float(default)


def _safe_int(z, key: str, default: int) -> int:
    if key not in z.files:
        return int(default)
    try:
        return int(np.array(z[key]).reshape(-1)[0])
    except Exception:
        return int(default)


def _pelvis_center(joints_xy: np.ndarray) -> np.ndarray:
    # MediaPipe hips: 23 (L), 24 (R). If missing, fall back to joint 23.
    V = joints_xy.shape[1]
    if V > 24:
        return 0.5 * (joints_xy[:, 23:24, :] + joints_xy[:, 24:25, :])
    if V > 23:
        return joints_xy[:, 23:24, :]
    return joints_xy[:, 0:1, :]


def _derive_mask(joints_xy: np.ndarray, conf: Optional[np.ndarray], conf_gate: float) -> np.ndarray:
    finite = np.isfinite(joints_xy[..., 0]) & np.isfinite(joints_xy[..., 1])
    if conf is None or conf_gate <= 0:
        return finite
    return finite & (conf >= float(conf_gate))


def _compute_motion(joints_xy: np.ndarray, fps: float, scale_by_fps: bool) -> np.ndarray:
    m = np.zeros_like(joints_xy, dtype=np.float32)
    m[1:] = joints_xy[1:] - joints_xy[:-1]
    m[0] = 0.0
    if scale_by_fps and fps > 0:
        m = m * float(fps)  # per-second velocity
    return m.astype(np.float32, copy=False)


def read_window_npz(path: str, fps_default: float = 30.0) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float, WindowMeta]:
    """Read raw arrays from a window NPZ. Returns (joints_xy, motion_xy, conf, mask, fps, meta)."""
    with np.load(path, allow_pickle=False) as z:
        if "joints" in z.files:
            joints = np.array(z["joints"], dtype=np.float32, copy=False)
        elif "xy" in z.files:
            joints = np.array(z["xy"], dtype=np.float32, copy=False)
        else:
            raise KeyError(f"Missing joints/xy in {path}")

        motion = np.array(z["motion"], dtype=np.float32, copy=False) if "motion" in z.files else None

        conf = None
        if "conf" in z.files:
            conf = np.array(z["conf"], dtype=np.float32, copy=False)
        elif "joints_conf" in z.files:
            conf = np.array(z["joints_conf"], dtype=np.float32, copy=False)

        mask = None
        if "mask" in z.files:
            mask = np.array(z["mask"], copy=False)
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = mask.astype(bool, copy=False)

        fps = _safe_scalar(z, "fps", fps_default)

        # meta
        video_id = _as_str(z["video_id"]) if "video_id" in z.files else (
            _as_str(z["seq_id"]) if "seq_id" in z.files else (
                _as_str(z["seq_stem"]) if "seq_stem" in z.files else (
                    _as_str(z["stem"]) if "stem" in z.files else ""
                )
            )
        )
        w_start = _safe_int(z, "w_start", 0)
        w_end = _safe_int(z, "w_end", w_start + joints.shape[0] - 1)
        y = _safe_int(z, "y", -1)
        meta = WindowMeta(path=path, video_id=video_id or "", w_start=w_start, w_end=w_end, fps=float(fps), y=int(y))

    return joints, motion, conf, mask, float(fps), meta


def build_gcn_input(
    joints_xy: np.ndarray,
    motion_xy: Optional[np.ndarray],
    conf: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    fps: float,
    feat_cfg: FeatCfg,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, mask_used) where X is [T,V,F] float32 and mask_used is [T,V] bool."""
    joints_xy = np.nan_to_num(joints_xy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if conf is not None:
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # mask
    if feat_cfg.use_precomputed_mask and mask is not None:
        m = mask.astype(bool, copy=False)
    else:
        m = _derive_mask(joints_xy, conf, feat_cfg.conf_gate)

    # centering
    if feat_cfg.center == "pelvis":
        joints_xy = joints_xy - _pelvis_center(joints_xy)

    # motion
    if feat_cfg.use_motion:
        if motion_xy is None or feat_cfg.center == "pelvis":
            motion_xy = _compute_motion(joints_xy, fps, feat_cfg.motion_scale_by_fps)
        else:
            motion_xy = np.nan_to_num(motion_xy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            if feat_cfg.motion_scale_by_fps and fps > 0:
                motion_xy = motion_xy * float(fps)
    else:
        motion_xy = np.zeros_like(joints_xy, dtype=np.float32)

    # apply mask to xy/motion; conf to zero out invalid joints
    mm = m.astype(np.float32)[..., None]
    joints_xy = joints_xy * mm
    if motion_xy is not None:
        motion_xy = motion_xy * mm
    if conf is not None:
        conf = conf * m.astype(np.float32)

    parts = [joints_xy]  # [T,V,2]
    if feat_cfg.use_motion:
        parts.append(motion_xy)
    if feat_cfg.use_conf_channel:
        if conf is None:
            conf = np.ones((joints_xy.shape[0], joints_xy.shape[1]), dtype=np.float32)
        parts.append(conf[..., None].astype(np.float32))
    X = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
    return X, m


def build_tcn_input(
    joints_xy: np.ndarray,
    motion_xy: Optional[np.ndarray],
    conf: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    fps: float,
    feat_cfg: FeatCfg,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, mask_used) where X is [T,C] float32."""
    Xg, m = build_gcn_input(joints_xy, motion_xy, conf, mask, fps, feat_cfg)
    T, V, F = Xg.shape
    # Flatten joints/features into channels. Conf channel stays 1 per joint (not duplicated).
    # For TCN we keep a stable ordering: [xy, (motion), (conf)]
    # where xy/motion are V*2 each and conf is V.
    xy = Xg[..., 0:2].reshape(T, V * 2)
    cur = [xy]
    if feat_cfg.use_motion:
        motion = Xg[..., 2:4].reshape(T, V * 2)
        cur.append(motion)
    if feat_cfg.use_conf_channel:
        conf1 = Xg[..., -1].reshape(T, V)
        cur.append(conf1)
    Xt = np.concatenate(cur, axis=1).astype(np.float32, copy=False)
    return Xt, m
