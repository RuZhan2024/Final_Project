#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
windows/make_unlabeled_windows.py

Create fixed-length windows for UNLABELED sequences (y = -1).

When to use
-----------
This is for "deployment-like" evaluation where you have no annotations,
e.g., LE2i Office/Lecture room sequences you want to replay later.

Input
-----
Sequence NPZ files from preprocess stage, expected keys:
  - xy   : [T,J,2]
  - conf : [T,J]
Optional:
  - mask : [T,J] (precomputed joint validity mask)

Output (per window)
-------------------
  <out_dir>/<subset>/<stem>__w{start:06d}_{end:06d}.npz

Keys saved:
  - xy/conf (legacy)
  - joints/motion/mask (model-ready)
  - y = -1, label = -1
  - meta fields: fps, video_id/seq_id/src/seq_stem, w_start/w_end
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


# ============================================================
# 1) IO helpers
# ============================================================
def atomic_save_npz(out_path: str, payload: Dict[str, Any]) -> None:
    """
    Atomic write:
      - write to out_path.tmp
      - then replace out_path

    Prevents half-written NPZs if process is interrupted.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    # IMPORTANT: write via file handle so NumPy does NOT auto-append ".npz"
    # when tmp does not end with ".npz" (e.g. ".npz.tmp").
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, str(p))


def read_stems(stems_txt: str) -> List[str]:
    """
    Read stems from a text file, one per line.
    Lines starting with # are treated as comments.
    """
    out: List[str] = []
    with open(stems_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def index_npz(npz_root: str) -> Dict[str, str]:
    """
    Index NPZ files recursively:
      {stem -> file_path}

    If duplicates exist (same stem appears twice in different folders),
    we keep the first and warn — duplicates are usually a data-layout bug.
    """
    idx: Dict[str, str] = {}
    files = sorted(glob.glob(os.path.join(npz_root, "**", "*.npz"), recursive=True))
    # Ignore temp artifacts (e.g. "*.tmp.npz" or "*.npz.tmp.npz") that can appear
    # if a previous run was interrupted mid-write.
    files = [p for p in files if ".tmp" not in pathlib.Path(p).name]

    dup = 0
    for p in files:
        stem = pathlib.Path(p).stem
        if stem in idx:
            dup += 1
            continue
        idx[stem] = p

    if dup:
        print(f"[WARN] duplicate stems in {npz_root}: {dup} (kept first occurrence)")
    return idx


def safe_fps(z: np.lib.npyio.NpzFile, fps_default: float) -> float:
    """Read fps from NPZ if present, else fallback."""
    if "fps" in z.files:
        try:
            return float(np.array(z["fps"]).reshape(-1)[0])
        except Exception:
            return float(fps_default)
    return float(fps_default)


# ============================================================
# 2) Feature helpers
# ============================================================
def derive_mask(xy: np.ndarray, conf: np.ndarray, conf_gate: float) -> np.ndarray:
    """
    Derive a joint-valid mask from xy/conf:
      valid = finite(xy) AND (conf >= conf_gate)

    Shape:
      xy   [W,J,2]
      conf [W,J]
      mask [W,J] boolean
    """
    finite = np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])
    if conf_gate > 0:
        return (conf >= conf_gate) & finite
    return finite


def compute_motion(joints: np.ndarray) -> np.ndarray:
    """
    Simple per-frame motion feature:
      motion[t] = joints[t] - joints[t-1]
      motion[0] = 0

    Shape:
      joints [W,J,2]
      motion [W,J,2]
    """
    motion = np.zeros_like(joints, dtype=np.float32)
    motion[1:] = joints[1:] - joints[:-1]
    motion[0] = 0.0
    return motion


def window_passes_quality(mask_w: np.ndarray, conf_w: np.ndarray, min_valid_frac: float, min_avg_conf: float) -> bool:
    """
    Quality gate:
      - min_avg_conf: reject if mean(conf) too low
      - min_valid_frac: reject if mean(mask) too low
    """
    if min_avg_conf > 0 and float(conf_w.mean()) < float(min_avg_conf):
        return False
    if min_valid_frac > 0 and float(mask_w.mean()) < float(min_valid_frac):
        return False
    return True


# ============================================================
# 3) Main
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()

    # Inputs
    ap.add_argument("--npz_dir", required=True, help="Processed sequence NPZ root")
    ap.add_argument("--stems_txt", required=True, help="Text file listing stems to windowize")
    ap.add_argument("--out_dir", required=True, help="Output windows root directory")

    # Window geometry
    ap.add_argument("--W", type=int, required=True, help="Window length (frames)")
    ap.add_argument("--stride", type=int, required=True, help="Step between window starts (frames)")

    # Metadata defaults
    ap.add_argument("--fps_default", type=float, default=30.0)
    ap.add_argument("--subset", default="test_unlabeled", help="Output subfolder name")

    # Reproducibility
    ap.add_argument("--seed", type=int, default=33724876)

    # Caps
    ap.add_argument("--max_windows_per_video", type=int, default=400, help="Cap windows per video (0 disables).")

    # Quality gating
    ap.add_argument("--use_precomputed_mask", action="store_true")
    ap.add_argument("--conf_gate", type=float, default=0.20)
    ap.add_argument("--min_valid_frac", type=float, default=0.00)
    ap.add_argument("--min_avg_conf", type=float, default=0.00)

    # Output behavior
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    if args.W <= 0 or args.stride <= 0:
        raise SystemExit("[ERR] W and stride must be positive integers.")

    rng = random.Random(int(args.seed))

    stems = read_stems(args.stems_txt)
    if not stems:
        raise SystemExit(f"[ERR] No stems found in {args.stems_txt}")

    idx = index_npz(args.npz_dir)

    out_base = os.path.join(args.out_dir, args.subset)
    os.makedirs(out_base, exist_ok=True)

    total_win = 0
    missing: List[str] = []
    too_short = 0
    skipped_quality = 0

    # ------------------------------------------------------------
    # For each requested stem, load sequence + generate windows
    # ------------------------------------------------------------
    for stem in stems:
        p = idx.get(stem)
        if p is None:
            missing.append(stem)
            continue

        # Load sequence data
        with np.load(p, allow_pickle=False) as z:
            if "xy" not in z.files or "conf" not in z.files:
                continue

            # Replace NaN with 0 so downstream never gets NaN explosions
            xy = np.nan_to_num(z["xy"]).astype(np.float32, copy=False)
            conf = np.nan_to_num(z["conf"]).astype(np.float32, copy=False)
            fps = safe_fps(z, args.fps_default)

            # Optional metadata (best-effort)
            seq_id = str(stem)
            src = ""
            seq_stem = str(stem)
            if "seq_id" in z.files:
                seq_id = str(np.array(z["seq_id"]).reshape(-1)[0])
            if "src" in z.files:
                src = str(np.array(z["src"]).reshape(-1)[0])
            if "seq_stem" in z.files:
                seq_stem = str(np.array(z["seq_stem"]).reshape(-1)[0])

            # Optional precomputed mask from preprocess stage
            pre_mask = None
            if args.use_precomputed_mask and "mask" in z.files:
                pre_mask = np.array(z["mask"])  # materialize before closing NPZ

        T = int(xy.shape[0])
        if T < args.W:
            too_short += 1
            continue

        # Generate all possible window starts
        starts = list(range(0, T - args.W + 1, int(args.stride)))

        # Optional cap (random sample for fairness + determinism)
        if args.max_windows_per_video > 0 and len(starts) > args.max_windows_per_video:
            starts = rng.sample(starts, args.max_windows_per_video)
            starts = sorted(starts)

        # --------------------------------------------------------
        # Save each window as one NPZ file
        # --------------------------------------------------------
        for st in starts:
            ed = st + args.W - 1  # inclusive end index (used in filename + metadata)

            out_name = f"{stem}__w{st:06d}_{ed:06d}.npz"
            out_path = os.path.join(out_base, out_name)

            if args.skip_existing and os.path.exists(out_path):
                continue

            xy_w = xy[st : ed + 1]
            conf_w = conf[st : ed + 1]

            # Choose mask source:
            # - if preprocess wrote mask and user enables it, trust that
            # - else derive from conf + finite(xy)
            if pre_mask is not None:
                mask_w = np.array(pre_mask[st : ed + 1]).astype(bool, copy=False)
            else:
                mask_w = derive_mask(xy_w, conf_w, float(args.conf_gate))

            if not window_passes_quality(mask_w, conf_w, float(args.min_valid_frac), float(args.min_avg_conf)):
                skipped_quality += 1
                continue

            joints = xy_w.astype(np.float32, copy=False)
            motion = compute_motion(joints)
            valid_frac = float(mask_w.mean()) if mask_w.size else 0.0

            payload = dict(
                # legacy keys
                xy=joints,
                conf=conf_w,

                # unlabeled signals
                y=np.int64(-1),
                label=np.int64(-1),

                # model-ready keys
                joints=joints,
                motion=motion,
                mask=mask_w.astype(np.uint8),

                # quality meta
                valid_frac=np.float32(valid_frac),
                overlap_frames=np.int16(0),
                overlap_frac=np.float32(0.0),

                # sequence meta
                fps=np.float32(fps),
                video_id=np.array(seq_id),
                seq_id=np.array(seq_id),
                src=np.array(src),
                seq_stem=np.array(seq_stem),

                # window meta
                w_start=np.int32(st),
                w_end=np.int32(ed),
            )

            atomic_save_npz(out_path, payload)
            total_win += 1

    print(f"[OK] wrote {total_win} unlabeled windows → {out_base}")
    if too_short:
        print(f"[WARN] {too_short} sequences shorter than W={args.W} (skipped).")
    if skipped_quality:
        print(f"[WARN] skipped_quality={skipped_quality}")
    if missing:
        print(f"[WARN] {len(missing)} stems not found in {args.npz_dir}. Examples: {missing[:5]}")


if __name__ == "__main__":
    main()
