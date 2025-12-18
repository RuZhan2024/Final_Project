
"""
preprocess_pose_npz.py

Clean + normalize MediaPipe Pose NPZ files BEFORE windowing.

Why this script exists
----------------------
Your current extractors write per-sequence/per-video pose files (xy/conf/fps),
and your trainers apply confidence gating inside the Dataset __getitem__.
That helps, but for supervisor-ready robustness you usually want a deterministic
preprocessing stage that you can run once and keep fixed:

  - jitter reduction (motion blur / fast motion)
  - gap filling (short missing stretches)
  - body-centric normalization (reduce scale/camera-distance shift)

This script reads NPZ files with:
  - xy   : [T, 33, 2] float32  (normalized 0..1, or NaNs/zeros when missing)
  - conf : [T, 33]    float32  (visibility; 0 means "missing")
  - fps  : optional scalar
and writes cleaned NPZs with the SAME keys (xy/conf/fps/size/src if present),
so downstream windowing/training can just point to the new directory.

Usage
-----
python preprocess_pose_npz.py \
  --in_dir  data/interim/le2i/pose_npz \
  --out_dir data/interim/le2i/pose_npz_clean \
  --recursive \
  --conf_thr 0.2 \
  --smooth_k 5 \
  --max_gap  4 \
  --normalize torso

Notes
-----
- "normalize torso" does:
    pelvis_center = mean(LeftHip(23), RightHip(24))
    shoulder_center = mean(LeftShoulder(11), RightShoulder(12))
    scale = median(||shoulder_center - pelvis_center||) over valid frames
    xy := (xy - pelvis_center) / max(scale, eps)

- If a joint is missing (conf ~ 0), we treat xy as missing even if it is 0.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


# MediaPipe Pose landmark indices used for normalization
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24


def list_npz(in_dir: str, recursive: bool) -> list[Path]:
    root = Path(in_dir)
    if recursive:
        return sorted(root.rglob("*.npz"))
    return sorted(root.glob("*.npz"))


def _as_missing_where_conf_zero(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """If conf==0, mark xy as NaN so later steps can treat it as missing."""
    xy2 = xy.copy()
    m = conf <= 0.0
    # m: [T,J] -> [T,J,1]
    xy2[m, :] = np.nan
    return xy2


def linear_fill_small_gaps(xy: np.ndarray, conf: np.ndarray, conf_thr: float, max_gap: int) -> np.ndarray:
    """
    Fill short missing gaps (<= max_gap frames) by linear interpolation,
    per joint and per coordinate. Uses conf>=conf_thr as "valid".
    """
    T, J, _ = xy.shape
    out = xy.copy()

    valid = (conf >= conf_thr) & np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])  # [T,J]

    for j in range(J):
        v = valid[:, j]
        if v.sum() < 2:
            continue

        for c in range(2):
            s = out[:, j, c]
            idx = np.arange(T)

            good = v & np.isfinite(s)
            if good.sum() < 2:
                continue

            # Find missing segments
            miss = ~good
            if not miss.any():
                continue

            # Identify contiguous missing runs
            run_starts = np.where(miss & ~np.r_[False, miss[:-1]])[0]
            run_ends = np.where(miss & ~np.r_[miss[1:], False])[0]

            for a, b in zip(run_starts, run_ends):
                gap_len = b - a + 1
                if gap_len > max_gap:
                    continue
                # Need valid neighbors on both sides for interpolation
                left = a - 1
                right = b + 1
                if left < 0 or right >= T:
                    continue
                if not good[left] or not good[right]:
                    continue
                s[a:right] = np.interp(idx[a:right], [left, right], [s[left], s[right]])

            out[:, j, c] = s

    return out


def smooth_weighted_moving_average(xy: np.ndarray, conf: np.ndarray, conf_thr: float, k: int) -> np.ndarray:
    """
    Weighted moving average smoothing across time:
      smoothed[t] = sum(w * x) / sum(w)
    where w = conf if conf>=conf_thr else 0.

    k should be odd (we will force odd).
    """
    if k <= 1:
        return xy
    if k % 2 == 0:
        k += 1

    T, J, C = xy.shape
    out = np.empty_like(xy)

    weights = np.where(conf >= conf_thr, conf, 0.0).astype(np.float32)  # [T,J]
    half = k // 2

    # pad with edge values (and pad weights similarly)
    xy_pad = np.pad(xy, ((half, half), (0, 0), (0, 0)), mode="edge")
    w_pad = np.pad(weights, ((half, half), (0, 0)), mode="edge")

    for t in range(T):
        w_win = w_pad[t:t + k]               # [k,J]
        x_win = xy_pad[t:t + k]             # [k,J,2]

        # If xy has NaNs, treat them as missing by zeroing their weight
        nan_mask = ~np.isfinite(x_win[..., 0]) | ~np.isfinite(x_win[..., 1])  # [k,J]
        w_eff = w_win.copy()
        w_eff[nan_mask] = 0.0

        denom = w_eff.sum(axis=0) + 1e-8     # [J]
        for c in range(C):
            num = (x_win[..., c] * w_eff).sum(axis=0)  # [J]
            out[t, :, c] = num / denom

    return out


def normalize_body_centric(
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
    mode: str,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Translate to pelvis center and scale by torso length (default) or shoulder width.

    Returns (xy_norm, meta).
    """
    mode = mode.lower()
    if mode == "none":
        return xy, {"norm": "none"}

    T, J, _ = xy.shape

    def joint_center(a: int, b: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return center position and a validity mask [T]."""
        pa, pb = xy[:, a, :], xy[:, b, :]
        va = conf[:, a] >= conf_thr
        vb = conf[:, b] >= conf_thr
        # if both valid, mean; if one valid, use it; else NaN
        center = np.full((T, 2), np.nan, np.float32)
        both = va & vb
        only_a = va & ~vb
        only_b = vb & ~va
        center[both] = 0.5 * (pa[both] + pb[both])
        center[only_a] = pa[only_a]
        center[only_b] = pb[only_b]
        valid = both | only_a | only_b
        return center, valid

    pelvis, pelvis_ok = joint_center(L_HIP, R_HIP)
    shoulders, sh_ok = joint_center(L_SHO, R_SHO)

    # Per-frame scale candidates
    if mode == "torso":
        d = np.linalg.norm(shoulders - pelvis, axis=1)  # [T]
        valid_d = pelvis_ok & sh_ok & np.isfinite(d)
    elif mode == "shoulder":
        d = np.linalg.norm(xy[:, L_SHO, :] - xy[:, R_SHO, :], axis=1)
        valid_d = (conf[:, L_SHO] >= conf_thr) & (conf[:, R_SHO] >= conf_thr) & np.isfinite(d)
    else:
        raise ValueError(f"Unknown normalize mode: {mode} (use none|torso|shoulder)")

    if valid_d.sum() == 0:
        scale = 1.0
    else:
        scale = float(np.median(d[valid_d]))
        if not np.isfinite(scale) or scale < eps:
            scale = 1.0

    # Translate by pelvis where available; if pelvis missing, subtract 0 (model will gate by conf anyway)
    pelvis_filled = np.nan_to_num(pelvis, nan=0.0, posinf=0.0, neginf=0.0)
    xy2 = xy - pelvis_filled[:, None, :]
    xy2 = xy2 / scale

    return xy2.astype(np.float32), {"norm": mode, "scale": float(scale), "conf_thr": float(conf_thr)}


def process_one(in_path: Path, out_path: Path, args) -> bool:
    with np.load(in_path, allow_pickle=False) as d:
        if "xy" not in d.files or "conf" not in d.files:
            return False
        xy = d["xy"].astype(np.float32)
        conf = d["conf"].astype(np.float32)

        meta = {}
        # carry through optional fields
        extras = {}
        for k in ("fps", "size", "src", "label", "y", "y_label", "target"):
            if k in d.files:
                extras[k] = d[k]

    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"{in_path}: expected xy [T,J,2], got {xy.shape}")

    # Standardise "missing": if conf==0 treat xy as missing even if it is [0,0]
    xy = _as_missing_where_conf_zero(xy, conf)

    # Fill short gaps then smooth
    if args.max_gap > 0:
        xy = linear_fill_small_gaps(xy, conf, conf_thr=args.conf_thr, max_gap=args.max_gap)

    if args.smooth_k > 1:
        xy = smooth_weighted_moving_average(xy, conf, conf_thr=args.conf_thr, k=args.smooth_k)

    # Normalise (translate+scale)
    xy, norm_meta = normalize_body_centric(xy, conf, conf_thr=args.conf_thr, mode=args.normalize)
    meta.update(norm_meta)

    # Replace NaNs with 0 for safe downstream (gating uses conf anyway)
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        xy=xy.astype(np.float32),
        conf=conf.astype(np.float32),
        preprocess=json.dumps(
            dict(
                conf_thr=float(args.conf_thr),
                smooth_k=int(args.smooth_k),
                max_gap=int(args.max_gap),
                normalize=str(args.normalize),
                **meta,
            )
        ),
        **extras,
    )
    return True


def parse_args():
    ap = argparse.ArgumentParser(description="Clean + normalise pose NPZs before windowing.")
    ap.add_argument("--in_dir", required=True, help="Directory containing pose NPZ files.")
    ap.add_argument("--out_dir", required=True, help="Where to write cleaned pose NPZ files.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    ap.add_argument("--conf_thr", type=float, default=0.2, help="Confidence threshold for 'valid' joints.")
    ap.add_argument("--smooth_k", type=int, default=5, help="Weighted moving-average window size (odd preferred).")
    ap.add_argument("--max_gap", type=int, default=4, help="Max missing gap (frames) to interpolate.")
    ap.add_argument("--normalize", choices=["none", "torso", "shoulder"], default="torso",
                    help="Body-centric normalization mode.")
    return ap.parse_args()


def main():
    args = parse_args()
    files = list_npz(args.in_dir, args.recursive)
    if not files:
        raise SystemExit(f"[ERR] no .npz files under: {args.in_dir}")

    in_root = Path(args.in_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    ok, fail = 0, 0

    for p in files:
        rel = p.resolve().relative_to(in_root)
        out_p = out_root / rel
        try:
            if process_one(p, out_p, args):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"[ERR] {p}: {e}")
            fail += 1

    print(f"[done] wrote {ok} cleaned files to {out_root}  (failed/skipped={fail})")


if __name__ == "__main__":
    main()
