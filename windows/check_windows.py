#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
windows/check_windows.py

Sanity-check window folders produced by:
- windows/make_windows.py
- windows/make_unlabeled_windows.py

What this script reports per split:
- total windows
- pos / neg / unlabeled / unknown / missing_y
- number of unique videos
- windows per video (mean / max)
- fraction of positives among labeled windows
- how many files contain each schema key: joints / motion / mask

Why it matters
--------------
This is your quick "did windowing work?" verification step.
Before training, you want to confirm:
- you have BOTH positive and negative windows (for labeled datasets)
- y exists and is correct
- schema keys expected by models exist

Usage:
  python windows/check_windows.py --root data/processed/le2i/windows_W32_S8
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np


# ============================================================
# 1) Small helper: read a "string-ish" value from NPZ
# ============================================================
def _as_str(v) -> str:
    """
    Convert different NPZ stored formats into a Python string safely.

    NPZ may store metadata as:
    - 0d numpy array: np.array("abc")
    - 1d array with 1 item: ["abc"]
    - bytes
    - plain Python string

    This helper standardizes them to a normal str.
    """
    try:
        if isinstance(v, np.ndarray):
            if v.shape == ():
                v = v.item()
            elif v.size == 1:
                v = v.reshape(-1)[0].item()
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="ignore")
        return str(v)
    except Exception:
        return str(v)


# ============================================================
# 2) Determine a "video key" for grouping windows
# ============================================================
def _video_key_from_npz(z: np.lib.npyio.NpzFile, fallback_path: str) -> str:
    """
    Prefer explicit metadata written by window builders:
      video_id, seq_id, seq_stem, src

    If missing, fallback to file name:
      <stem>__w000123_000155.npz  ->  <stem>
    """
    for k in ("video_id", "seq_id", "seq_stem", "src"):
        if k in z.files:
            s = _as_str(z[k])
            if s and s != "None":
                return s

    stem = pathlib.Path(fallback_path).stem
    if "__w" in stem:
        return stem.split("__w", 1)[0]
    return stem


# ============================================================
# 3) Count one split directory
# ============================================================
def count_split(split_dir: str) -> Tuple[int, int, int, int, int, int, int, float, float, int, int, int, int]:
    """
    Returns a fixed tuple so printing is easy.

    Output fields:
      total, pos, neg, unlabeled, unknown, missing_y,
      n_videos, mean_per_video, pos_frac, max_per_video,
      has_joints, has_motion, has_mask
    """
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    if not files:
        return (0, 0, 0, 0, 0, 0, 0, 0.0, float("nan"), 0, 0, 0, 0)

    total = pos = neg = unl = unk = missing_y = 0
    has_joints = has_motion = has_mask = 0

    per_video: Dict[str, int] = defaultdict(int)

    for p in files:
        try:
            # IMPORTANT: allow_pickle=False (safer) — our NPZ schema does not require pickle.
            with np.load(p, allow_pickle=False) as z:
                total += 1

                # Schema presence checks
                if "joints" in z.files:
                    has_joints += 1
                if "motion" in z.files:
                    has_motion += 1
                if "mask" in z.files:
                    has_mask += 1

                # Label parsing
                if "y" not in z.files:
                    missing_y += 1
                    y = None
                else:
                    try:
                        y = int(np.array(z["y"]).reshape(-1)[0])
                    except Exception:
                        y = None
                        missing_y += 1

                # Label buckets
                if y is None:
                    pass
                elif y == 1:
                    pos += 1
                elif y == 0:
                    neg += 1
                elif y == -1:
                    unl += 1
                else:
                    unk += 1

                # Video grouping
                vid = _video_key_from_npz(z, p)
                per_video[vid] += 1

        except Exception:
            # Broken file or unreadable NPZ — we skip it silently to keep scan robust.
            continue

    n_videos = len(per_video)
    mean_per_video = (sum(per_video.values()) / max(1, n_videos)) if total else 0.0
    max_per_video = max(per_video.values()) if per_video else 0

    # pos_frac is computed only over labeled windows (pos+neg), unlabeled are ignored
    pos_frac = (pos / max(1, pos + neg)) if (pos + neg) > 0 else float("nan")

    return (
        total,
        pos,
        neg,
        unl,
        unk,
        missing_y,
        n_videos,
        float(mean_per_video),
        float(pos_frac),
        int(max_per_video),
        int(has_joints),
        int(has_motion),
        int(has_mask),
    )


# ============================================================
# 4) Main CLI
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Windows root containing train/val/test/... subfolders")
    args = ap.parse_args()

    splits = ["train", "val", "test", "unsplit", "test_unlabeled"]

    print(f"[scan] {args.root}")
    for s in splits:
        d = os.path.join(args.root, s)
        if not os.path.isdir(d):
            continue

        (
            n,
            p,
            n0,
            unl,
            unk,
            my,
            vids,
            mean_v,
            pos_frac,
            max_v,
            hj,
            hm,
            hk,
        ) = count_split(d)

        print(
            f"{s:15s} files={n:6d}  pos={p:6d}  neg={n0:6d}  unl={unl:6d}  "
            f"unk={unk:6d}  missing_y={my:6d}  videos={vids:5d}  "
            f"win/video≈{mean_v:7.2f}  max/video={max_v:5d}  pos_frac={pos_frac:6.3f}  "
            f"schema(joints/motion/mask)={hj:6d}/{hm:6d}/{hk:6d}"
        )


if __name__ == "__main__":
    main()
