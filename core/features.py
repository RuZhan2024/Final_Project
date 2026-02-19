#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/features.py

Canonical feature builder + layout helpers.

Contract B (repo-wide):
  - A window feature tensor is always [T, V, F].
  - The exact channel ordering is defined by channel_layout(feat_cfg).
  - No module should slice by hard-coded indices; always use channel_layout.

Window NPZ schema (preferred):
  - joints : [T,V,2] float32
  - motion : [T,V,2] float32 (optional)
  - conf   : [T,V]   float32 (optional)
  - mask   : [T,V]   bool/uint8 (optional)
  - fps    : scalar  (optional)
  - y      : 0/1 (labeled) or -1 (unlabeled)

Older/compat keys:
  - xy instead of joints
  - joints_conf instead of conf

Features implemented (all per-joint):
  - xy            (always present)
  - motion_xy     (optional)
  - bone_xy       (optional)      joint - parent(joint)
  - bone_len      (optional)      ||bone_xy||
  - conf          (optional)

Note:
  - build_canonical_input() returns X[T,V,F] and mask[T,V] (bool).
  - split_gcn_two_stream() returns (X_joint, X_motion) using the layout.
  - build_tcn_input() flattens to [T, V*F] for TCN.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------- config + meta ----------------

@dataclass(frozen=True)
class FeatCfg:
    center: str = "pelvis"  # "pelvis" | "none"

    # Core channels
    use_motion: bool = True
    use_bone: bool = False
    use_bone_length: bool = False
    use_conf_channel: bool = True

    # Motion normalization
    motion_scale_by_fps: bool = True

    # Masking
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
            use_bone=bool(d.get("use_bone", False)),
            use_bone_length=bool(d.get("use_bone_length", False)),
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


# ---------------- small helpers ----------------


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


def _pelvis_center(joints_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask-aware pelvis center.

    MediaPipe hips: 23 (L), 24 (R). When hips are missing, fall back to a
    robust torso center using any available shoulders/hips.

    Returns: center[T,1,2] float32.
    """
    joints_xy = joints_xy.astype(np.float32, copy=False)
    m = mask.astype(bool, copy=False)

    T, V = int(joints_xy.shape[0]), int(joints_xy.shape[1])

    def _masked_mean(idxs):
        idxs = [int(i) for i in idxs if int(i) < V]
        if not idxs:
            return np.zeros((T, 1, 2), dtype=np.float32), np.zeros((T,), dtype=bool)

        sub = joints_xy[:, idxs, :]                     # [T,K,2]
        vm = m[:, idxs].astype(np.float32, copy=False)  # [T,K]
        cnt = vm.sum(axis=1)                            # [T]
        s = (sub * vm[..., None]).sum(axis=1)           # [T,2]

        out = np.zeros((T, 2), dtype=np.float32)
        ok = cnt > 0
        out[ok] = s[ok] / cnt[ok, None]
        return out[:, None, :], ok

    # Preferred: average hips when valid.
    if V > 24:
        lh = joints_xy[:, 23, :]
        rh = joints_xy[:, 24, :]
        vl = m[:, 23]
        vr = m[:, 24]

        center = np.zeros((T, 2), dtype=np.float32)
        ok = vl | vr
        both = vl & vr
        center[both] = 0.5 * (lh[both] + rh[both])
        only_l = vl & ~vr
        only_r = vr & ~vl
        center[only_l] = lh[only_l]
        center[only_r] = rh[only_r]

        # Fallback for frames with no valid hips: average shoulders/hips.
        if (~ok).any():
            fb, fb_ok = _masked_mean([11, 12, 23, 24])
            use = (~ok) & fb_ok
            center[use] = fb[:, 0, :][use]

        return center[:, None, :].astype(np.float32, copy=False)

    # Fallbacks for other skeleton layouts.
    fb, _ = _masked_mean([23, 11, 12] if V > 23 else [11, 12])
    return fb


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


# ---------------- layout ----------------


def channel_layout(cfg: FeatCfg) -> Dict[str, Tuple[int, int]]:
    """Return a dict of {name: (start, stop)} slices for X[..., start:stop]."""
    i = 0
    out: Dict[str, Tuple[int, int]] = {}

    out["xy"] = (i, i + 2)
    i += 2

    if bool(cfg.use_motion):
        out["motion"] = (i, i + 2)
        i += 2

    if bool(cfg.use_bone):
        out["bone"] = (i, i + 2)
        i += 2

    if bool(cfg.use_bone_length):
        out["bone_len"] = (i, i + 1)
        i += 1

    if bool(cfg.use_conf_channel):
        out["conf"] = (i, i + 1)
        i += 1

    out["F"] = (0, i)
    return out


def feature_dim_per_joint(cfg: FeatCfg) -> int:
    """Total feature dimension per joint (F in X[T,V,F])."""
    lo = channel_layout(cfg)
    return int(lo["F"][1] - lo["F"][0])


# ---------------- bones ----------------


def _default_parents(V: int) -> np.ndarray:
    """Return a parent index array of length V.

    - For MediaPipe Pose 33 and COCO 17, uses a reasonable tree.
    - For unknown V, falls back to parent[i]=max(i-1,0).

    Parent of a root/self joint is itself.
    """
    V = int(V)
    if V == 33:
        p = list(range(33))
        # Face
        p[0] = 0
        p[1] = 0; p[2] = 1; p[3] = 2
        p[4] = 0; p[5] = 4; p[6] = 5
        p[7] = 3; p[8] = 6
        p[9] = 0; p[10] = 0
        # Upper body
        p[11] = 0; p[12] = 0
        p[13] = 11; p[14] = 12
        p[15] = 13; p[16] = 14
        p[17] = 15; p[18] = 16
        p[19] = 15; p[20] = 16
        p[21] = 15; p[22] = 16
        # Lower body
        p[23] = 11; p[24] = 12
        p[25] = 23; p[26] = 24
        p[27] = 25; p[28] = 26
        p[29] = 27; p[30] = 28
        p[31] = 29; p[32] = 30
        return np.asarray(p, dtype=np.int64)

    if V == 17:
        p = list(range(17))
        p[0] = 0
        p[1] = 0; p[2] = 0
        p[3] = 1; p[4] = 2
        p[5] = 0; p[6] = 0
        p[7] = 5; p[8] = 6
        p[9] = 7; p[10] = 8
        p[11] = 5; p[12] = 6
        p[13] = 11; p[14] = 12
        p[15] = 13; p[16] = 14
        return np.asarray(p, dtype=np.int64)

    p = np.arange(V, dtype=np.int64)
    p[0] = 0
    for i in range(1, V):
        p[i] = i - 1
    return p


def _compute_bones(joints_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bone vectors + lengths for a sequence.

    Args:
      joints_xy: [T,V,2]

    Returns:
      bone_xy:  [T,V,2]
      bone_len: [T,V]
    """
    joints_xy = np.asarray(joints_xy, dtype=np.float32)
    T, V, _ = joints_xy.shape
    parents = _default_parents(V)

    parent_xy = joints_xy[:, parents, :]  # [T,V,2]
    bone_xy = joints_xy - parent_xy

    bone_len = np.sqrt(np.sum(bone_xy ** 2, axis=-1) + 1e-12).astype(np.float32, copy=False)
    return bone_xy.astype(np.float32, copy=False), bone_len


# ---------------- IO ----------------


def read_window_npz(
    path: str,
    fps_default: float = 30.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float, WindowMeta]:
    """Read raw arrays from a window NPZ.

    Returns:
      joints_xy, motion_xy, conf, mask, fps, meta
    """
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
        y = _safe_int(z, "y", _safe_int(z, "label", -1))

        meta = WindowMeta(
            path=path,
            video_id=video_id or "",
            w_start=int(w_start),
            w_end=int(w_end),
            fps=float(fps),
            y=int(y),
        )

    return joints, motion, conf, mask, float(fps), meta


# ---------------- builders ----------------


def build_canonical_input(
    joints_xy: np.ndarray,
    motion_xy: Optional[np.ndarray],
    conf: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    fps: float,
    feat_cfg: FeatCfg,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build canonical X[T,V,F] and mask[T,V] from raw arrays."""
    joints_xy = np.asarray(joints_xy, dtype=np.float32)
    T, V, _ = joints_xy.shape

    # Choose a mask (precomputed mask preferred, otherwise derive from conf/finite).
    if bool(feat_cfg.use_precomputed_mask) and mask is not None:
        m = np.asarray(mask, dtype=bool)
    else:
        m = _derive_mask(joints_xy, conf, feat_cfg.conf_gate)

    # Build a conf array for the conf-channel.
    conf_ch: Optional[np.ndarray] = None
    if bool(feat_cfg.use_conf_channel):
        if conf is None:
            conf_ch = m.astype(np.float32)
        else:
            conf_ch = np.asarray(conf, dtype=np.float32)
            if conf_ch.shape != (T, V):
                conf_ch = conf_ch.reshape(T, V)
            conf_ch = np.where(m, conf_ch, 0.0).astype(np.float32, copy=False)

    # Centering
    xy = joints_xy.copy()
    if str(feat_cfg.center) == "pelvis":
        xy = xy - _pelvis_center(xy, m)

    # Apply mask by zeroing invalid joints.
    xy = np.where(m[..., None], xy, 0.0).astype(np.float32, copy=False)

    # Motion
    mot: Optional[np.ndarray] = None
    if bool(feat_cfg.use_motion):
        if motion_xy is None:
            mot = _compute_motion(xy, float(fps), bool(feat_cfg.motion_scale_by_fps))
        else:
            mot = np.asarray(motion_xy, dtype=np.float32)
            if mot.shape != xy.shape:
                mot = mot.reshape(xy.shape)
            if bool(feat_cfg.motion_scale_by_fps) and float(fps) > 0:
                mot = mot * float(fps)
        mot = np.where(m[..., None], mot, 0.0).astype(np.float32, copy=False)

    # Bones
    bone_xy: Optional[np.ndarray] = None
    bone_len: Optional[np.ndarray] = None
    if bool(feat_cfg.use_bone) or bool(feat_cfg.use_bone_length):
        bone_xy0, bone_len0 = _compute_bones(xy)
        # Mask: if a joint is invalid, its bone features are treated as zero.
        bone_xy0 = np.where(m[..., None], bone_xy0, 0.0)
        bone_len0 = np.where(m, bone_len0, 0.0)
        if bool(feat_cfg.use_bone):
            bone_xy = bone_xy0.astype(np.float32, copy=False)
        if bool(feat_cfg.use_bone_length):
            bone_len = bone_len0.astype(np.float32, copy=False)

    # Concatenate channels in the canonical order.
    feats = [xy]
    if mot is not None:
        feats.append(mot)
    if bone_xy is not None:
        feats.append(bone_xy)
    if bone_len is not None:
        feats.append(bone_len[..., None])
    if conf_ch is not None:
        feats.append(conf_ch[..., None])

    X = np.concatenate(feats, axis=-1).astype(np.float32, copy=False)
    return X, m.astype(bool, copy=False)


def split_gcn_two_stream(X: np.ndarray, feat_cfg: FeatCfg) -> Tuple[np.ndarray, np.ndarray]:
    """Split canonical X[T,V,F] into (joint_stream, motion_stream).

    Joint stream includes: xy (+ bone/bone_len) (+ conf)
    Motion stream includes: motion_xy if enabled, otherwise zeros.
    """
    X = np.asarray(X, dtype=np.float32)
    T, V, F = X.shape

    lo = channel_layout(feat_cfg)

    joint_parts = [X[..., slice(*lo["xy"])]]
    if "bone" in lo:
        joint_parts.append(X[..., slice(*lo["bone"])])
    if "bone_len" in lo:
        joint_parts.append(X[..., slice(*lo["bone_len"])])
    if "conf" in lo:
        joint_parts.append(X[..., slice(*lo["conf"])])

    xj = np.concatenate(joint_parts, axis=-1).astype(np.float32, copy=False)

    if "motion" in lo:
        xm = X[..., slice(*lo["motion"])].astype(np.float32, copy=False)
    else:
        xm = np.zeros((T, V, 2), dtype=np.float32)

    return xj, xm


def build_tcn_input(X: np.ndarray, feat_cfg: FeatCfg) -> np.ndarray:
    """Flatten canonical X[T,V,F] into TCN input x[T, V*F]."""
    X = np.asarray(X, dtype=np.float32)
    T, V, F = X.shape
    _ = feat_cfg  # kept for signature symmetry; flatten includes all channels.
    return X.reshape(T, V * F).astype(np.float32, copy=False)
