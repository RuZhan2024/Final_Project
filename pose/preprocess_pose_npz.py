#!/usr/bin/env python3
"""
preprocess_pose_npz.py

Clean + normalize MediaPipe Pose NPZ files BEFORE windowing.

Why this rewrite?
-----------------
Your downstream windowing / models work best when the pose sequences:
  - have consistent "missing" semantics
  - have short gaps filled (optional)
  - are smoothed (optional)
  - are normalised in a body-centric coordinate system (optional)
  - include explicit validity masks (joint-level + frame-level)

Input NPZ (from your extractors)
--------------------------------
Required:
  - xy   : (T, 33, 2) float32  (often 0..1 image-normalised; may contain NaNs)
  - conf : (T, 33)    float32  (visibility/confidence; 0 means "missing")

Optional (carried through unchanged):
  - fps, size, src, seq_id, frames, etc.

Output NPZ
----------
Always writes:
  - xy, conf (same keys for backward compatibility)
  - mask        : (T, 33) uint8   1 if joint is valid (after fill, if enabled)
  - frame_mask  : (T,)    uint8   1 if frame has enough valid joints
  - valid_ratio : (T,)    float32 fraction of valid joints in frame
  - preprocess  : JSON string (parameters + summary)

Notes
-----
- We do NOT drop frames (to preserve alignment with span annotations).
  Instead, we provide frame_mask/valid_ratio and optionally zero-out
  frames that are too corrupted (see --invalidate_bad_frames).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# MediaPipe Pose landmark indices used for normalisation / rotation
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24


# -------------------------
# IO helpers
# -------------------------
def list_npz(in_dir: str, recursive: bool) -> list[Path]:
    root = Path(in_dir)
    return sorted(root.rglob("*.npz") if recursive else root.glob("*.npz"))


# -------------------------
# Missing-value handling
# -------------------------
def standardize_missing(xy: np.ndarray, conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make "missing" consistent:
      - non-finite conf -> 0
      - where conf <= 0 OR xy non-finite -> xy becomes NaN
    """
    xy2 = xy.astype(np.float32, copy=True)
    conf2 = conf.astype(np.float32, copy=True)

    conf2 = np.nan_to_num(conf2, nan=0.0, posinf=0.0, neginf=0.0)

    bad_xy = ~np.isfinite(xy2[..., 0]) | ~np.isfinite(xy2[..., 1])
    bad_conf = conf2 <= 0.0
    miss = bad_xy | bad_conf  # (T,J)

    # Set missing xy to NaN (both channels)
    xy2[miss] = np.nan
    conf2[bad_xy] = 0.0
    return xy2, conf2


# -------------------------
# Gap filling (xy + conf)
# -------------------------
def linear_fill_small_gaps(
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
    max_gap: int,
    fill_conf: str = "thr",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill short missing gaps (<= max_gap) by linear interpolation for each joint.

    Parameters
    ----------
    fill_conf:
      - "keep"         : keep conf as-is (filled xy still has low conf)
      - "thr"          : set filled conf to conf_thr
      - "min_neighbors": set to min(conf[left], conf[right])
      - "linear"       : linear interpolation for conf too

    Returns
    -------
    xy_out, conf_out, filled_mask  where filled_mask is (T,J) bool.
    """
    if max_gap <= 0:
        return xy, conf, np.zeros(conf.shape, dtype=bool)

    fill_conf = fill_conf.lower()
    if fill_conf not in ("keep", "thr", "min_neighbors", "linear"):
        raise ValueError("fill_conf must be one of: keep|thr|min_neighbors|linear")

    T, J, _ = xy.shape
    out_xy = xy.copy()
    out_conf = conf.copy()
    filled = np.zeros((T, J), dtype=bool)

    valid = (conf >= conf_thr) & np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])  # (T,J)

    idx = np.arange(T)

    for j in range(J):
        v = valid[:, j]
        if v.sum() < 2:
            continue

        # For xy channels
        for c in range(2):
            s = out_xy[:, j, c]
            good = v & np.isfinite(s)
            if good.sum() < 2:
                continue

            miss = ~good
            if not miss.any():
                continue

            run_starts = np.where(miss & ~np.r_[False, miss[:-1]])[0]
            run_ends = np.where(miss & ~np.r_[miss[1:], False])[0]

            for a, b in zip(run_starts, run_ends):
                gap_len = b - a + 1
                if gap_len > max_gap:
                    continue

                left = a - 1
                right = b + 1
                if left < 0 or right >= T:
                    continue
                if not good[left] or not good[right]:
                    continue

                # Fill xy
                s[a:right] = np.interp(idx[a:right], [left, right], [s[left], s[right]])
                filled[a:right, j] = True

            out_xy[:, j, c] = s

        # Optionally fill conf where we filled xy
        if filled[:, j].any() and fill_conf != "keep":
            miss_j = filled[:, j]

            if fill_conf == "thr":
                out_conf[miss_j, j] = np.maximum(out_conf[miss_j, j], conf_thr).astype(np.float32)
            else:
                # Determine runs again to compute neighbors
                miss = miss_j
                run_starts = np.where(miss & ~np.r_[False, miss[:-1]])[0]
                run_ends = np.where(miss & ~np.r_[miss[1:], False])[0]

                for a, b in zip(run_starts, run_ends):
                    left = a - 1
                    right = b + 1
                    if left < 0 or right >= T:
                        continue

                    cl = float(out_conf[left, j])
                    cr = float(out_conf[right, j])

                    if fill_conf == "min_neighbors":
                        val = float(min(cl, cr))
                        out_conf[a:right, j] = max(val, conf_thr)
                    elif fill_conf == "linear":
                        out_conf[a:right, j] = np.interp(idx[a:right], [left, right], [cl, cr]).astype(np.float32)

    return out_xy, out_conf, filled


# -------------------------
# Smoothing
# -------------------------
def smooth_weighted_moving_average(xy: np.ndarray, conf: np.ndarray, conf_thr: float, k: int) -> np.ndarray:
    """
    Weighted moving average smoothing.
    If a joint has no valid samples in the window, output remains NaN.
    """
    if k <= 1:
        return xy
    if k % 2 == 0:
        k += 1

    T, J, C = xy.shape
    out = np.full_like(xy, np.nan, dtype=np.float32)

    w = np.where((conf >= conf_thr) & np.isfinite(conf), conf, 0.0).astype(np.float32)  # (T,J)
    half = k // 2

    xy_pad = np.pad(xy, ((half, half), (0, 0), (0, 0)), mode="edge")
    w_pad = np.pad(w, ((half, half), (0, 0)), mode="edge")

    for t in range(T):
        w_win = w_pad[t : t + k]   # (k,J)
        x_win = xy_pad[t : t + k]  # (k,J,C)

        nan_mask = ~np.isfinite(x_win[..., 0]) | ~np.isfinite(x_win[..., 1])  # (k,J)
        w_eff = w_win.copy()
        w_eff[nan_mask] = 0.0

        denom = w_eff.sum(axis=0)  # (J,)
        ok = denom > 1e-8
        if not ok.any():
            continue

        for c in range(C):
            num = (x_win[..., c] * w_eff).sum(axis=0)  # (J,)
            out[t, ok, c] = num[ok] / denom[ok]

    return out


# -------------------------
# Normalisation / rotation
# -------------------------
def _fill_nearest_2d(center: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid frames in center[T,2] with nearest valid frame (in time)."""
    T = center.shape[0]
    valid2 = valid & np.isfinite(center[:, 0]) & np.isfinite(center[:, 1])
    if valid2.sum() == 0:
        return np.zeros_like(center, dtype=np.float32)

    idx = np.arange(T)
    last_valid = np.where(valid2, idx, -1)
    last = np.maximum.accumulate(last_valid)

    next_valid = np.where(valid2, idx, T)
    nxt = np.minimum.accumulate(next_valid[::-1])[::-1]

    choose = np.empty(T, dtype=np.int64)
    for t in range(T):
        if last[t] == -1:
            choose[t] = nxt[t]
        elif nxt[t] == T:
            choose[t] = last[t]
        else:
            choose[t] = last[t] if (t - last[t]) <= (nxt[t] - t) else nxt[t]

    out = center.copy().astype(np.float32)
    bad = ~valid2
    out[bad] = center[choose[bad]]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _joint_center(xy: np.ndarray, conf: np.ndarray, a: int, b: int, conf_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return center[T,2] and valid[T] using joints a/b with fallback if one missing."""
    T = xy.shape[0]
    pa, pb = xy[:, a, :], xy[:, b, :]
    va = conf[:, a] >= conf_thr
    vb = conf[:, b] >= conf_thr

    center = np.full((T, 2), np.nan, np.float32)
    both = va & vb
    only_a = va & ~vb
    only_b = vb & ~va
    center[both] = 0.5 * (pa[both] + pb[both])
    center[only_a] = pa[only_a]
    center[only_b] = pb[only_b]
    valid = both | only_a | only_b
    return center, valid


def _rotation_angle_shoulders(xy: np.ndarray, conf: np.ndarray, conf_thr: float) -> float:
    """
    Compute a stable (sequence-level) rotation angle using shoulder vector.
    We rotate by -angle so that the shoulder line becomes horizontal.
    """
    ls = xy[:, L_SHO, :]
    rs = xy[:, R_SHO, :]
    ok = (conf[:, L_SHO] >= conf_thr) & (conf[:, R_SHO] >= conf_thr)
    ok = ok & np.isfinite(ls[:, 0]) & np.isfinite(ls[:, 1]) & np.isfinite(rs[:, 0]) & np.isfinite(rs[:, 1])
    if ok.sum() == 0:
        return 0.0
    v = rs[ok] - ls[ok]
    ang = np.arctan2(v[:, 1], v[:, 0])  # radians
    # Use median to reduce jitter/outliers
    a = float(np.median(ang))
    if not np.isfinite(a):
        return 0.0
    return a


def normalize_body_centric(
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
    mode: str,
    pelvis_fill: str = "nearest",
    rotate: str = "none",
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Translate to pelvis center and scale by torso length or shoulder width.
    Optional: rotate so shoulders are horizontal (sequence-level).
    """
    mode = mode.lower()
    rotate = rotate.lower()
    if mode == "none" and rotate == "none":
        return xy, {"norm": "none", "rot": "none"}

    if rotate not in ("none", "shoulders"):
        raise ValueError("rotate must be one of: none|shoulders")

    T = xy.shape[0]

    pelvis, pelvis_ok = _joint_center(xy, conf, L_HIP, R_HIP, conf_thr)
    shoulders, sh_ok = _joint_center(xy, conf, L_SHO, R_SHO, conf_thr)

    if mode == "none":
        scale = 1.0
    elif mode == "torso":
        d = np.linalg.norm(shoulders - pelvis, axis=1)
        valid_d = pelvis_ok & sh_ok & np.isfinite(d)
        scale = float(np.median(d[valid_d])) if valid_d.sum() else 1.0
    elif mode == "shoulder":
        d = np.linalg.norm(xy[:, L_SHO, :] - xy[:, R_SHO, :], axis=1)
        valid_d = (conf[:, L_SHO] >= conf_thr) & (conf[:, R_SHO] >= conf_thr) & np.isfinite(d)
        scale = float(np.median(d[valid_d])) if valid_d.sum() else 1.0
    else:
        raise ValueError(f"Unknown normalize mode: {mode} (use none|torso|shoulder)")

    if (not np.isfinite(scale)) or scale < eps:
        scale = 1.0

    if pelvis_fill == "nearest":
        pelvis_used = _fill_nearest_2d(pelvis, pelvis_ok)
    else:
        pelvis_used = np.nan_to_num(pelvis, nan=0.0, posinf=0.0, neginf=0.0)

    xy2 = xy.astype(np.float32, copy=False)

    # Translate and scale
    if mode != "none":
        xy2 = (xy2 - pelvis_used[:, None, :]) / float(scale)
    else:
        xy2 = (xy2 - pelvis_used[:, None, :])

    # Rotate around origin (already pelvis-centered)
    rot_angle = 0.0
    if rotate == "shoulders":
        rot_angle = _rotation_angle_shoulders(xy, conf, conf_thr)
        ca = float(np.cos(-rot_angle))
        sa = float(np.sin(-rot_angle))
        R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
        xy2 = xy2 @ R.T

    meta = {
        "norm": mode,
        "scale": float(scale),
        "rot": rotate,
        "rot_angle": float(rot_angle),
        "conf_thr": float(conf_thr),
    }
    return xy2.astype(np.float32), meta


# -------------------------
# Masks / frame quality
# -------------------------
def compute_masks(xy: np.ndarray, conf: np.ndarray, conf_thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      joint_mask  : (T,J) bool
      frame_mask  : (T,)  bool  (>= min_valid_ratio handled outside)
      valid_ratio : (T,)  float32
    """
    joint_mask = (conf >= conf_thr) & np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])
    valid_ratio = joint_mask.mean(axis=1).astype(np.float32)
    frame_mask = np.isfinite(valid_ratio)  # refined later
    return joint_mask, frame_mask, valid_ratio


def invalidate_bad_frames(
    xy: np.ndarray,
    conf: np.ndarray,
    joint_mask: np.ndarray,
    frame_mask: np.ndarray,
    valid_ratio: np.ndarray,
    min_valid_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    If a frame has too few valid joints, mark it invalid and zero-out xy/conf.
    This preserves indexing while preventing bad frames from harming downstream.
    """
    if min_valid_ratio <= 0.0:
        return xy, conf, frame_mask

    fm = frame_mask.copy()
    bad = valid_ratio < float(min_valid_ratio)
    if not bad.any():
        return xy, conf, fm

    fm[bad] = False
    xy2 = xy.copy()
    conf2 = conf.copy()

    xy2[bad, :, :] = 0.0
    conf2[bad, :] = 0.0
    joint_mask[bad, :] = False
    return xy2, conf2, fm


# -------------------------
# Main per-file processing
# -------------------------
def process_one(in_path: Path, out_path: Path, args) -> bool:
    if args.skip_existing and out_path.exists():
        return True

    with np.load(in_path, allow_pickle=False) as d:
        if "xy" not in d.files or "conf" not in d.files:
            return False

        xy = d["xy"].astype(np.float32)
        conf = d["conf"].astype(np.float32)

        # Carry through everything else (future-proof)
        extras = {k: d[k] for k in d.files if k not in ("xy", "conf", "preprocess", "mask", "frame_mask", "valid_ratio")}

    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"{in_path}: expected xy (T,J,2), got {xy.shape}")
    if conf.ndim != 2:
        raise ValueError(f"{in_path}: expected conf (T,J), got {conf.shape}")
    if xy.shape[0] != conf.shape[0] or xy.shape[1] != conf.shape[1]:
        raise ValueError(f"{in_path}: xy {xy.shape} and conf {conf.shape} mismatch")

    # 1) standardize missing semantics
    xy, conf = standardize_missing(xy, conf)

    # 2) fill short gaps
    filled_mask = np.zeros(conf.shape, dtype=bool)
    if args.max_gap > 0:
        xy, conf, filled_mask = linear_fill_small_gaps(
            xy, conf, conf_thr=args.conf_thr, max_gap=args.max_gap, fill_conf=args.fill_conf
        )

    # 3) smooth
    if args.smooth_k > 1:
        xy = smooth_weighted_moving_average(xy, conf, conf_thr=args.conf_thr, k=args.smooth_k)

    # 4) normalise + optional rotation
    xy, norm_meta = normalize_body_centric(
        xy,
        conf,
        conf_thr=args.conf_thr,
        mode=args.normalize,
        pelvis_fill=args.pelvis_fill,
        rotate=args.rotate,
    )

    # 5) masks / frame quality
    joint_mask, frame_mask, valid_ratio = compute_masks(xy, conf, conf_thr=args.conf_thr)

    if args.invalidate_bad_frames:
        xy, conf, frame_mask = invalidate_bad_frames(
            xy, conf, joint_mask, frame_mask, valid_ratio, min_valid_ratio=args.min_valid_ratio
        )

    # Final: replace NaNs with 0 for safe downstream IO
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Summary for reproducibility
    preprocess_meta = dict(
        conf_thr=float(args.conf_thr),
        smooth_k=int(args.smooth_k),
        max_gap=int(args.max_gap),
        fill_conf=str(args.fill_conf),
        normalize=str(args.normalize),
        rotate=str(args.rotate),
        pelvis_fill=str(args.pelvis_fill),
        min_valid_ratio=float(args.min_valid_ratio),
        invalidate_bad_frames=bool(args.invalidate_bad_frames),
        frames=int(xy.shape[0]),
        joints=int(xy.shape[1]),
        filled_points=int(filled_mask.sum()),
        valid_ratio_mean=float(np.nanmean(valid_ratio)) if valid_ratio.size else 0.0,
        valid_ratio_min=float(np.nanmin(valid_ratio)) if valid_ratio.size else 0.0,
        **{k: v for k, v in norm_meta.items() if k != "conf_thr"},
    )

    np.savez_compressed(
        out_path,
        xy=xy,
        conf=conf,
        mask=joint_mask.astype(np.uint8),
        frame_mask=frame_mask.astype(np.uint8),
        valid_ratio=valid_ratio.astype(np.float32),
        preprocess=json.dumps(preprocess_meta),
        **extras,
    )
    return True


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Clean + normalise pose NPZs before windowing.")
    ap.add_argument("--in_dir", required=True, help="Directory containing pose NPZ files.")
    ap.add_argument("--out_dir", required=True, help="Where to write cleaned pose NPZ files.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    ap.add_argument("--skip_existing", action="store_true", help="Skip writing if output already exists.")

    ap.add_argument("--conf_thr", type=float, default=0.2, help="Confidence threshold for 'valid' joints.")
    ap.add_argument("--smooth_k", type=int, default=5, help="Weighted moving-average window size (odd preferred).")

    ap.add_argument("--max_gap", type=int, default=4, help="Max missing gap (frames) to interpolate.")
    ap.add_argument(
        "--fill_conf",
        choices=["keep", "thr", "min_neighbors", "linear"],
        default="thr",
        help="How to set conf for interpolated points (default: thr).",
    )

    ap.add_argument(
        "--normalize",
        choices=["none", "torso", "shoulder"],
        default="torso",
        help="Body-centric normalization mode.",
    )
    ap.add_argument(
        "--rotate",
        choices=["none", "shoulders"],
        default="none",
        help="Optional rotation: align shoulders horizontally (sequence-level).",
    )
    ap.add_argument(
        "--pelvis_fill",
        choices=["nearest", "zero"],
        default="nearest",
        help="How to fill missing pelvis center before translation.",
    )

    ap.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.25,
        help="Minimum fraction of valid joints for a frame to be considered usable.",
    )
    ap.add_argument(
        "--invalidate_bad_frames",
        action="store_true",
        help="If set, frames with valid_ratio < min_valid_ratio are zeroed out + flagged in frame_mask.",
    )

    ap.add_argument("--log_every", type=int, default=200, help="Print progress every N files (default 200).")
    return ap.parse_args()


def main():
    args = parse_args()
    files = list_npz(args.in_dir, args.recursive)
    if not files:
        raise SystemExit(f"[ERR] no .npz files under: {args.in_dir}")

    in_root = Path(args.in_dir).resolve()
    out_root = Path(args.out_dir).resolve()

    ok, fail = 0, 0
    for i, p in enumerate(files, start=1):
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

        if args.log_every > 0 and (i % args.log_every == 0):
            print(f"[prog] {i}/{len(files)}  ok={ok}  fail={fail}")

    print(f"[done] wrote {ok} cleaned files to {out_root}  (failed/skipped={fail})")


if __name__ == "__main__":
    main()
