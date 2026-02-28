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
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np

_TORSO4_IDX = np.asarray([11, 12, 23, 24], dtype=np.int64)
_FALLBACK_PELVIS_IDX = [23, 11, 12]
_FALLBACK_SHOULDERS_IDX = [11, 12]


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


def _masked_center_from_indices(joints_xy: np.ndarray, mask: np.ndarray, idxs: list[int]) -> np.ndarray:
    """Return mask-aware center[T,1,2] for the selected joint indices."""
    T, V, _ = joints_xy.shape
    use = [int(i) for i in idxs if 0 <= int(i) < V]
    if not use:
        return np.zeros((T, 1, 2), dtype=np.float32)
    sub = joints_xy[:, use, :]                         # [T,K,2]
    vm = mask[:, use].astype(np.float32, copy=False)  # [T,K]
    cnt = vm.sum(axis=1)                               # [T]
    weighted = np.empty_like(sub, dtype=np.float32)
    np.multiply(sub, vm[..., None], out=weighted)
    num = weighted.sum(axis=1)            # [T,2]
    out = np.zeros((T, 2), dtype=np.float32)
    ok = cnt > 0.0
    if bool(np.any(ok)):
        out[ok] = num[ok] / cnt[ok, None]
    return out[:, None, :]


def _pelvis_center(joints_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask-aware pelvis center.

    MediaPipe hips: 23 (L), 24 (R). When hips are missing, fall back to a
    robust torso center using any available shoulders/hips.

    Returns: center[T,1,2] float32.
    """
    if not (isinstance(joints_xy, np.ndarray) and joints_xy.dtype == np.float32):
        joints_xy = np.asarray(joints_xy, dtype=np.float32)
    if isinstance(mask, np.ndarray) and mask.dtype == np.bool_:
        m = mask
    else:
        m = np.asarray(mask, dtype=bool)

    T, V = int(joints_xy.shape[0]), int(joints_xy.shape[1])

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
            miss_idx = np.flatnonzero(~ok).astype(np.int64, copy=False)
            sub = np.take(joints_xy[miss_idx], _TORSO4_IDX, axis=1)  # [M,4,2]
            vm = np.take(m[miss_idx], _TORSO4_IDX, axis=1).astype(np.float32, copy=False)  # [M,4]
            cnt = vm.sum(axis=1)  # [M]
            use = cnt > 0
            if bool(np.any(use)):
                weighted = np.empty_like(sub, dtype=np.float32)
                np.multiply(sub, vm[..., None], out=weighted)
                num = weighted.sum(axis=1)  # [M,2]
                center[miss_idx[use]] = num[use] / cnt[use, None]

        return center[:, None, :]

    # Fallbacks for other skeleton layouts.
    return _masked_center_from_indices(
        joints_xy,
        m,
        _FALLBACK_PELVIS_IDX if V > 23 else _FALLBACK_SHOULDERS_IDX,
    )


def _derive_mask(joints_xy: np.ndarray, conf: Optional[np.ndarray], conf_gate: float) -> np.ndarray:
    finite = np.isfinite(joints_xy[..., 0]) & np.isfinite(joints_xy[..., 1])
    if conf is None or conf_gate <= 0:
        return finite
    return finite & (conf >= float(conf_gate))


def _compute_motion(joints_xy: np.ndarray, fps: float, scale_by_fps: bool) -> np.ndarray:
    if not (isinstance(joints_xy, np.ndarray) and joints_xy.dtype == np.float32):
        joints_xy = np.asarray(joints_xy, dtype=np.float32)
    if joints_xy.shape[0] == 0:
        return np.empty_like(joints_xy, dtype=np.float32)
    m = np.empty_like(joints_xy, dtype=np.float32)
    np.subtract(joints_xy[1:], joints_xy[:-1], out=m[1:])
    m[0] = 0.0
    if scale_by_fps and fps > 0:
        m *= float(fps)  # per-second velocity
    return m


# ---------------- layout ----------------


def channel_layout(cfg: FeatCfg) -> Dict[str, Tuple[int, int]]:
    """Return a dict of {name: (start, stop)} slices for X[..., start:stop]."""
    return dict(
        _channel_layout_cached(
            bool(cfg.use_motion),
            bool(cfg.use_bone),
            bool(cfg.use_bone_length),
            bool(cfg.use_conf_channel),
        )
    )


@lru_cache(maxsize=16)
def _channel_layout_cached(
    use_motion: bool,
    use_bone: bool,
    use_bone_length: bool,
    use_conf_channel: bool,
) -> Tuple[Tuple[str, Tuple[int, int]], ...]:
    """Cached channel layout tuple to reduce repeated dict rebuilds."""
    i = 0
    out: Dict[str, Tuple[int, int]] = {}

    out["xy"] = (i, i + 2)
    i += 2

    if use_motion:
        out["motion"] = (i, i + 2)
        i += 2

    if use_bone:
        out["bone"] = (i, i + 2)
        i += 2

    if use_bone_length:
        out["bone_len"] = (i, i + 1)
        i += 1

    if use_conf_channel:
        out["conf"] = (i, i + 1)
        i += 1

    out["F"] = (0, i)
    return tuple(out.items())


def feature_dim_per_joint(cfg: FeatCfg) -> int:
    """Total feature dimension per joint (F in X[T,V,F])."""
    lo = channel_layout(cfg)
    return int(lo["F"][1] - lo["F"][0])


# ---------------- bones ----------------


@lru_cache(maxsize=8)
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
    if V > 1:
        p[1:] -= 1
    return p


def _compute_bones(joints_xy: np.ndarray, *, compute_len: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute bone vectors and (optionally) lengths for a sequence.

    Args:
      joints_xy: [T,V,2]

    Returns:
      bone_xy:  [T,V,2]
      bone_len: [T,V] or None when compute_len=False
    """
    if not (isinstance(joints_xy, np.ndarray) and joints_xy.dtype == np.float32):
        joints_xy = np.asarray(joints_xy, dtype=np.float32)
    T, V, _ = joints_xy.shape
    parents = _default_parents(V)

    # `np.take(..., axis=1)` is faster than advanced indexing for this hot path.
    parent_xy = np.take(joints_xy, parents, axis=1)  # [T,V,2]
    bone_xy = np.empty_like(joints_xy, dtype=np.float32)
    np.subtract(joints_xy, parent_xy, out=bone_xy)

    bone_len: Optional[np.ndarray]
    if compute_len:
        # `hypot` is faster than sum+sqrt for 2D vectors and keeps numerical stability.
        bone_len = np.empty((T, V), dtype=np.float32)
        np.hypot(bone_xy[..., 0], bone_xy[..., 1], out=bone_len)
        np.maximum(bone_len, np.float32(1e-6), out=bone_len)
    else:
        bone_len = None
    return bone_xy, bone_len


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
    *,
    assume_finite_xy: bool = False,
    assume_finite_conf: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build canonical X[T,V,F] and mask[T,V] from raw arrays."""
    if isinstance(joints_xy, np.ndarray) and joints_xy.dtype == np.float32:
        joints_xy = joints_xy
    else:
        joints_xy = np.asarray(joints_xy, dtype=np.float32)
    if joints_xy.ndim != 3 or joints_xy.shape[-1] != 2:
        raise ValueError(f"joints_xy must have shape [T,V,2], got {tuple(joints_xy.shape)}")
    T, V, _ = joints_xy.shape
    use_motion = bool(feat_cfg.use_motion)
    use_bone = bool(feat_cfg.use_bone)
    use_bone_len = bool(feat_cfg.use_bone_length)
    use_conf_ch = bool(feat_cfg.use_conf_channel)
    conf_gate_f = float(feat_cfg.conf_gate)
    conf_arr_opt: Optional[np.ndarray] = None
    conf_fin: Optional[np.ndarray] = None
    conf_fin_all: Optional[bool] = None
    if conf is not None:
        if isinstance(conf, np.ndarray) and conf.dtype == np.float32:
            conf_arr_opt = conf
        else:
            conf_arr_opt = np.asarray(conf, dtype=np.float32)
        if conf_arr_opt.shape != (T, V):
            if int(conf_arr_opt.size) != int(T * V):
                raise ValueError(
                    f"conf must have shape [T,V]=({T},{V}) or equivalent size {T*V}, got {tuple(conf_arr_opt.shape)}"
                )
            conf_arr_opt = conf_arr_opt.reshape(T, V)

    # Joint finite mask is used to guard against extractor noise/occlusion artifacts.
    if assume_finite_xy:
        finite_xy = None
    else:
        finite_xy = np.isfinite(joints_xy[..., 0]) & np.isfinite(joints_xy[..., 1])

    # Choose a mask (precomputed mask preferred, otherwise derive from conf/finite).
    if bool(feat_cfg.use_precomputed_mask) and mask is not None:
        if isinstance(mask, np.ndarray) and mask.dtype == np.bool_:
            m = mask
        else:
            m = np.asarray(mask, dtype=bool)
        if m.shape != (T, V):
            if int(m.size) != int(T * V):
                raise ValueError(
                    f"mask must have shape [T,V]=({T},{V}) or equivalent size {T*V}, got {tuple(m.shape)}"
                )
            m = m.reshape(T, V)
        # Never trust precomputed masks over invalid coordinates.
        if finite_xy is not None:
            m = m & finite_xy
    else:
        if conf is None or conf_gate_f <= 0.0:
            if finite_xy is None:
                m = np.ones((T, V), dtype=bool)
            else:
                m = finite_xy
        else:
            assert conf_arr_opt is not None
            if assume_finite_conf:
                conf_fin = None
                conf_fin_all = True
                if finite_xy is None:
                    m = conf_arr_opt >= conf_gate_f
                else:
                    m = finite_xy & (conf_arr_opt >= conf_gate_f)
            else:
                conf_fin = np.isfinite(conf_arr_opt)
                conf_fin_all = bool(conf_fin.all())
                if finite_xy is None:
                    m = conf_fin & (conf_arr_opt >= conf_gate_f)
                else:
                    m = finite_xy & conf_fin & (conf_arr_opt >= conf_gate_f)
    m_all = bool(m.all())
    m_f: Optional[np.ndarray] = None
    m_f3: Optional[np.ndarray] = None
    if not m_all:
        m_f = m.astype(np.float32, copy=False)
        m_f3 = m_f[..., None]

    # Reuse a contiguous float32 confidence buffer once when possible to avoid
    # repeated temporary arrays in the output assembly path.
    conf_arr_clean: Optional[np.ndarray] = None
    if use_conf_ch and conf_arr_opt is not None:
        conf_arr_clean = conf_arr_opt
        if conf_fin_all is None:
            if assume_finite_conf:
                conf_fin_all = True
            else:
                conf_fin = np.isfinite(conf_arr_opt)
                conf_fin_all = bool(conf_fin.all())
        if not bool(conf_fin_all):
            conf_arr_clean = np.nan_to_num(conf_arr_opt, nan=0.0, posinf=0.0, neginf=0.0)

    # Centering
    if str(feat_cfg.center) == "pelvis":
        # Common fast path for MediaPipe-33 with fully valid masks.
        if m_all and V > 24:
            center = 0.5 * (joints_xy[:, 23:24, :] + joints_xy[:, 24:25, :])
        else:
            center = _pelvis_center(joints_xy, m)
        xy = joints_xy - center
        xy_is_new = True
    else:
        xy = joints_xy
        xy_is_new = False

    # Apply mask by zeroing invalid joints.
    if not m_all:
        if not xy_is_new:
            xy = xy.copy()
        xy *= m_f3

    # Motion
    mot: Optional[np.ndarray] = None
    if use_motion:
        if motion_xy is None:
            mot = _compute_motion(xy, float(fps), bool(feat_cfg.motion_scale_by_fps))
            mot_is_new = True
        else:
            if isinstance(motion_xy, np.ndarray) and motion_xy.dtype == np.float32:
                mot = motion_xy
            else:
                mot = np.asarray(motion_xy, dtype=np.float32)
            if mot.shape != xy.shape:
                if int(mot.size) != int(xy.size):
                    raise ValueError(
                        f"motion_xy must match joints shape [T,V,2]={tuple(xy.shape)} or equivalent size {xy.size}, got {tuple(mot.shape)}"
                    )
                mot = mot.reshape(xy.shape)
            mot_is_new = False
            if bool(feat_cfg.motion_scale_by_fps) and float(fps) > 0:
                mot = mot * float(fps)
                mot_is_new = True
        if not m_all:
            if not mot_is_new:
                mot = mot.copy()
            mot *= m_f3

    # Bones
    bone_xy: Optional[np.ndarray] = None
    bone_len: Optional[np.ndarray] = None
    if use_bone or use_bone_len:
        bone_xy0, bone_len0 = _compute_bones(xy, compute_len=use_bone_len)
        # Mask: if a joint is invalid, its bone features are treated as zero.
        if not m_all:
            bone_xy0 *= m_f3
            if bone_len0 is not None:
                bone_len0 *= m_f
        if use_bone:
            bone_xy = bone_xy0
        if use_bone_len:
            assert bone_len0 is not None
            bone_len = bone_len0

    # Assemble directly into a preallocated output tensor to avoid temporary
    # arrays from concatenate() on the critical inference path.
    F = 2 + (2 if use_motion else 0) + (2 if use_bone else 0) + (1 if use_bone_len else 0) + (1 if use_conf_ch else 0)
    X = np.empty((T, V, F), dtype=np.float32)
    off = 0
    X[..., off:off + 2] = xy
    off += 2
    if mot is not None:
        X[..., off:off + 2] = mot
        off += 2
    if bone_xy is not None:
        X[..., off:off + 2] = bone_xy
        off += 2
    if bone_len is not None:
        X[..., off] = bone_len
        off += 1
    if use_conf_ch:
        if conf is None:
            if m_all:
                X[..., off] = 1.0
            else:
                X[..., off] = m_f
        else:
            assert conf_arr_clean is not None
            X[..., off] = conf_arr_clean
            if not m_all:
                X[..., off] *= m_f
    return X, m


def split_gcn_two_stream(X: np.ndarray, feat_cfg: FeatCfg) -> Tuple[np.ndarray, np.ndarray]:
    """Split canonical X[T,V,F] into (joint_stream, motion_stream).

    Joint stream includes: xy (+ bone/bone_len) (+ conf)
    Motion stream includes: motion_xy if enabled, otherwise zeros.
    """
    if isinstance(X, np.ndarray) and X.dtype == np.float32:
        X = X
    else:
        X = np.asarray(X, dtype=np.float32)
    T, V, F = X.shape

    lo = channel_layout(feat_cfg)
    xy_sl = slice(*lo["xy"])
    has_bone = "bone" in lo
    has_bone_len = "bone_len" in lo
    has_conf = "conf" in lo
    has_motion = "motion" in lo

    if not has_motion:
        # Fast path: no motion channel configured, so full tensor is joint stream.
        xj = X
    elif not has_bone and not has_bone_len and not has_conf:
        # Fast path: joint stream is xy only (view/no-op cast).
        xj = X[..., xy_sl]
    else:
        jF = 2 + (2 if has_bone else 0) + (1 if has_bone_len else 0) + (1 if has_conf else 0)
        xj = np.empty((T, V, jF), dtype=np.float32)
        off = 0
        xj[..., off:off + 2] = X[..., xy_sl]
        off += 2
        if has_bone:
            xj[..., off:off + 2] = X[..., slice(*lo["bone"])]
            off += 2
        if has_bone_len:
            xj[..., off:off + 1] = X[..., slice(*lo["bone_len"])]
            off += 1
        if has_conf:
            xj[..., off:off + 1] = X[..., slice(*lo["conf"])]

    if has_motion:
        xm = X[..., slice(*lo["motion"])]
    else:
        xm = np.zeros((T, V, 2), dtype=np.float32)

    return xj, xm


def build_tcn_input(X: np.ndarray, feat_cfg: FeatCfg) -> np.ndarray:
    """Flatten canonical X[T,V,F] into TCN input x[T, V*F]."""
    if isinstance(X, np.ndarray) and X.dtype == np.float32:
        X = X
    else:
        X = np.asarray(X, dtype=np.float32)
    T, V, F = X.shape
    _ = feat_cfg  # kept for signature symmetry; flatten includes all channels.
    return X.reshape(T, V * F)
