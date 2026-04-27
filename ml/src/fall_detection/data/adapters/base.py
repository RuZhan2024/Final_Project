#!/usr/bin/env python3
"""Dataset adapter layer for multi-dataset canonicalization.

This module is intentionally independent of model code. It converts dataset
sequence NPZ files into one stable payload contract that can be consumed by the
legacy canonical feature builder (`core.features.build_canonical_input`).
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


# Internal 17-joint order (COCO-17):
# [nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, l_elbow,
#  r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]
INTERNAL_17_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Map internal-17 index -> MediaPipe Pose-33 index.
MP33_TO_INTERNAL17: Tuple[int, ...] = (
    0,   # nose
    2,   # left_eye
    5,   # right_eye
    7,   # left_ear
    8,   # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
)


@dataclass
class AdapterOutput:
    """Canonical per-sequence payload used by legacy feature builders."""

    joints_xy: np.ndarray  # [T,V,2] float32 (V=33 for mp33, V=17 for internal17)
    motion_xy: Optional[np.ndarray]  # [T,V,2] float32 or None
    conf: Optional[np.ndarray]  # [T,V] float32 or None
    mask: Optional[np.ndarray]  # [T,V] bool or None
    fps: float
    meta: Dict[str, object]


def _safe_scalar(z: np.lib.npyio.NpzFile, key: str, default: float) -> float:
    if key not in z.files:
        return float(default)
    try:
        return float(np.array(z[key]).reshape(-1)[0])
    except Exception:
        return float(default)


def _safe_int(z: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    if key not in z.files:
        return int(default)
    try:
        return int(np.array(z[key]).reshape(-1)[0])
    except Exception:
        return int(default)


def _as_str(x: object) -> str:
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


def map_mp33_to_internal17(
    joints_xy: np.ndarray,
    conf: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Project MediaPipe-33 arrays onto the internal 17-joint layout."""
    xy = np.asarray(joints_xy, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"joints_xy must be [T,V,2], got {xy.shape}")
    if int(xy.shape[1]) < 29:
        raise ValueError(f"Expected MediaPipe-style joints with V>=29, got V={xy.shape[1]}")

    idx = np.asarray(MP33_TO_INTERNAL17, dtype=np.int64)
    xy17 = xy[:, idx, :].astype(np.float32, copy=False)

    conf17: Optional[np.ndarray] = None
    if conf is not None:
        c = np.asarray(conf, dtype=np.float32)
        conf17 = c[:, idx].astype(np.float32, copy=False)

    mask17: Optional[np.ndarray] = None
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 3:
            m = m[..., 0]
        mask17 = m[:, idx].astype(bool, copy=False)

    return xy17, conf17, mask17


def _resample_linear_1d(values: np.ndarray, t_old: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    out = np.empty((t_new.shape[0],), dtype=np.float32)
    out[:] = np.interp(t_new, t_old, values).astype(np.float32, copy=False)
    return out


def resample_temporal(
    joints_xy: np.ndarray,
    conf: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    fps_src: float,
    fps_dst: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Resample a sequence from fps_src to fps_dst along time axis."""
    if fps_src <= 0 or fps_dst <= 0 or abs(fps_src - fps_dst) < 1e-6:
        return joints_xy, conf, mask

    xy = np.asarray(joints_xy, dtype=np.float32)
    T, V, C = xy.shape
    if T <= 1:
        return xy, conf, mask

    duration_s = float(T - 1) / float(fps_src)
    t_old = np.linspace(0.0, duration_s, T, dtype=np.float32)
    T_new = int(max(1, round(duration_s * float(fps_dst)) + 1))
    t_new = np.linspace(0.0, duration_s, T_new, dtype=np.float32)

    xy_new = np.empty((T_new, V, C), dtype=np.float32)
    for j in range(V):
        for c in range(C):
            xy_new[:, j, c] = _resample_linear_1d(xy[:, j, c], t_old, t_new)

    conf_new: Optional[np.ndarray] = None
    if conf is not None:
        cf = np.asarray(conf, dtype=np.float32)
        conf_new = np.empty((T_new, V), dtype=np.float32)
        for j in range(V):
            conf_new[:, j] = _resample_linear_1d(cf[:, j], t_old, t_new)

    mask_new: Optional[np.ndarray] = None
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        nearest_idx = np.clip(np.rint(t_new * float(fps_src)).astype(np.int64), 0, T - 1)
        mask_new = m[nearest_idx, :].astype(bool, copy=False)

    return xy_new, conf_new, mask_new


class DatasetAdapter(ABC):
    """Base adapter API for sequence-level canonicalization."""

    dataset_name: str = ""
    target_fps: Optional[float] = None
    joint_layout: str = "mp33"  # "mp33" (default) | "internal17"

    def __init__(self, *, joint_layout: str = "mp33") -> None:
        jl = str(joint_layout).strip().lower()
        if jl not in {"mp33", "internal17"}:
            raise ValueError(f"Unsupported joint_layout '{joint_layout}'. Use 'mp33' or 'internal17'.")
        self.joint_layout = jl

    def iter_sequences(self, npz_dir: str, recursive: bool = True) -> Iterable[str]:
        root = Path(npz_dir)
        pat = "**/*.npz" if recursive else "*.npz"
        for p in sorted(root.glob(pat)):
            if p.is_file():
                yield str(p)

    def load_sequence(self, npz_path: str, fps_default: float = 30.0) -> AdapterOutput:
        with np.load(npz_path, allow_pickle=False) as z:
            if "joints" in z.files:
                joints = np.array(z["joints"], dtype=np.float32, copy=False)
            elif "xy" in z.files:
                joints = np.array(z["xy"], dtype=np.float32, copy=False)
            else:
                raise KeyError(f"Missing joints/xy in {npz_path}")

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
            w_start = _safe_int(z, "w_start", 0)
            w_end = _safe_int(z, "w_end", w_start + int(joints.shape[0]) - 1)
            y = _safe_int(z, "y", _safe_int(z, "label", -1))

            video_id = ""
            for key in ("video_id", "seq_id", "seq_stem", "stem"):
                if key in z.files:
                    video_id = _as_str(z[key])
                    break
            if not video_id:
                video_id = Path(npz_path).stem
            seq_id = _as_str(z["seq_id"]) if "seq_id" in z.files else video_id
            src = _as_str(z["src"]) if "src" in z.files else ""
            seq_stem = _as_str(z["seq_stem"]) if "seq_stem" in z.files else Path(npz_path).stem

        joints_out = joints.astype(np.float32, copy=False)
        conf_out = conf.astype(np.float32, copy=False) if conf is not None else None
        mask_out = mask.astype(bool, copy=False) if mask is not None else None

        if self.joint_layout == "internal17":
            joints_out, conf_out, mask_out = map_mp33_to_internal17(joints_out, conf=conf_out, mask=mask_out)

        if self.target_fps is not None:
            joints_out, conf_out, mask_out = resample_temporal(
                joints_xy=joints_out,
                conf=conf_out,
                mask=mask_out,
                fps_src=float(fps),
                fps_dst=float(self.target_fps),
            )
            fps = float(self.target_fps)
            w_end = int(w_start + joints_out.shape[0] - 1)

        return AdapterOutput(
            joints_xy=joints_out,
            motion_xy=None,
            conf=conf_out,
            mask=mask_out,
            fps=float(fps),
            meta={
                "path": str(npz_path),
                "video_id": str(video_id),
                "w_start": int(w_start),
                "w_end": int(w_end),
                "y": int(y),
                "dataset": self.dataset_name,
                "seq_id": str(seq_id),
                "src": str(src),
                "seq_stem": str(seq_stem),
            },
        )


class LE2iAdapter(DatasetAdapter):
    dataset_name = "le2i"
    target_fps = None


class CAUCAFallAdapter(DatasetAdapter):
    dataset_name = "caucafall"
    target_fps = None


class MUVIMAdapter(DatasetAdapter):
    dataset_name = "muvim"
    target_fps = None


class URFallAdapter(DatasetAdapter):
    dataset_name = "urfall"

    def __init__(self, target_fps: float = 25.0, *, joint_layout: str = "mp33") -> None:
        super().__init__(joint_layout=joint_layout)
        self.target_fps = float(target_fps)


def build_adapter(
    dataset: str,
    urfall_target_fps: float = 25.0,
    *,
    joint_layout: str = "mp33",
) -> DatasetAdapter:
    ds = str(dataset).strip().lower()
    if ds == "le2i":
        return LE2iAdapter(joint_layout=joint_layout)
    if ds == "caucafall":
        return CAUCAFallAdapter(joint_layout=joint_layout)
    if ds == "muvim":
        return MUVIMAdapter(joint_layout=joint_layout)
    if ds in ("urfall", "urfd"):
        return URFallAdapter(target_fps=urfall_target_fps, joint_layout=joint_layout)
    raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: le2i, caucafall, muvim, urfall")
