#!/usr/bin/env python3
"""windows/check_windows.py

Lightweight sanity-check for window folders produced by windows/make_windows.py
or windows/make_unlabeled_windows.py.

It prints per-split counts:
- total windows
- pos / neg / unlabeled / unknown / missing_y
- number of unique videos
- windows per video (mean / max)
- how many files contain each schema key: joints / motion / mask

Use:
  python windows/check_windows.py --root data/processed/le2i/windows_W32_S8
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
from collections import Counter, defaultdict
from typing import Dict, Tuple

import numpy as np


def _video_key_from_npz(z, fallback_path: str) -> str:
    # Prefer explicit metadata written by make_windows.py
    for k in ("video_id", "seq_id", "seq_stem", "src"):
        if k in z.files:
            try:
                v = z[k].item() if np.ndim(z[k]) == 0 else z[k]
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="ignore")
                v = str(v)
                if v and v != "None":
                    return v
            except Exception:
                pass

    # Fallback: derive from filename '<stem>__w000123_000155'
    stem = pathlib.Path(fallback_path).stem
    if "__w" in stem:
        return stem.split("__w", 1)[0]
    return stem


def count_split(split_dir: str) -> Tuple[int, int, int, int, int, int, int, float, float, int, int, int, int]:
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    if not files:
        return (0, 0, 0, 0, 0, 0, 0, 0.0, float("nan"), 0, 0, 0, 0)

    total = 0
    pos = 0
    neg = 0
    unl = 0
    unk = 0
    missing_y = 0

    has_joints = 0
    has_motion = 0
    has_mask = 0

    per_video: Dict[str, int] = defaultdict(int)
    for p in files:
        try:
            z = np.load(p, allow_pickle=True)
        except Exception:
            continue

        total += 1
        if "joints" in z.files:
            has_joints += 1
        if "motion" in z.files:
            has_motion += 1
        if "mask" in z.files:
            has_mask += 1

        if "y" not in z.files:
            missing_y += 1
            y = None
        else:
            try:
                y = int(np.array(z["y"]).reshape(-1)[0])
            except Exception:
                y = None
                missing_y += 1

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

        vid = _video_key_from_npz(z, p)
        per_video[vid] += 1

    n_videos = len(per_video)
    mean_per_video = (sum(per_video.values()) / max(1, n_videos)) if total else 0.0
    max_per_video = max(per_video.values()) if per_video else 0
    pos_frac = (pos / max(1, pos + neg)) if (pos + neg) > 0 else float("nan")

    return (
        total,
        pos,
        neg,
        unl,
        unk,
        missing_y,
        n_videos,
        mean_per_video,
        pos_frac,
        max_per_video,
        has_joints,
        has_motion,
        has_mask,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="windows root, containing train/val/test/... subfolders")
    args = ap.parse_args()

    splits = ["train", "val", "test", "unsplit", "test_unlabeled"]
    print(f"[scan] {args.root}")
    for s in splits:
        d = os.path.join(args.root, s)
        if not os.path.isdir(d):
            continue
        n, p, n0, unl, unk, my, vids, mean_v, pos_frac, max_v, hj, hm, hk = count_split(d)
        print(
            f"{s:15s} files={n:6d}  pos={p:6d}  neg={n0:6d}  unl={unl:6d}  "
            f"unk={unk:6d}  missing_y={my:6d}  videos={vids:5d}  "
            f"win/video≈{mean_v:7.2f}  max/video={max_v:5d}  pos_frac={pos_frac:6.3f}  "
            f"schema(joints/motion/mask)={hj:6d}/{hm:6d}/{hk:6d}"
        )


if __name__ == "__main__":
    main()
