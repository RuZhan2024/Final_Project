#!/usr/bin/env python3
"""
windows/check_windows.py  (updated)

Scan a windows directory (train/val/test/unsplit/test_unlabeled) and report:
- total files
- positives (y==1)
- negatives (y==0)
- unlabeled (y==-1)
- unknown (other y)
- missing y key
- schema presence: joints/motion/mask keys
- unique videos (from video_id/seq_id/stem)
- windows per video (mean / max)

This is compatible with the redesigned window schema.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np


def _safe_npz_str(z, key: str) -> str | None:
    if key not in z.files:
        return None
    try:
        v = np.array(z[key]).reshape(-1)[0]
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        return str(v)
    except Exception:
        return None


def count_split(path: str) -> Tuple[int, int, int, int, int, int, int, int, float, int]:
    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    pos = neg = unl = unk = miss = 0

    has_joints = has_motion = has_mask = 0

    per_video: Dict[str, int] = defaultdict(int)

    for p in files:
        try:
            with np.load(p, allow_pickle=False) as z:
                if "y" not in z.files:
                    miss += 1
                    continue
                y = int(np.array(z["y"]).reshape(()))

                if "joints" in z.files:
                    has_joints += 1
                if "motion" in z.files:
                    has_motion += 1
                if "mask" in z.files:
                    has_mask += 1

                vid = (
                    _safe_npz_str(z, "video_id")
                    or _safe_npz_str(z, "seq_id")
                    or _safe_npz_str(z, "stem")
                    or _safe_npz_str(z, "seq_stem")
                )
                if not vid:
                    vid = os.path.splitext(os.path.basename(p))[0]

        except Exception:
            miss += 1
            continue

        per_video[vid] += 1

        if y == 1:
            pos += 1
        elif y == 0:
            neg += 1
        elif y == -1:
            unl += 1
        else:
            unk += 1

    n = len(files)
    unique_videos = len(per_video)
    mean_per_video = (n / unique_videos) if unique_videos else 0.0
    max_per_video = max(per_video.values()) if per_video else 0
    pos_frac = (pos / (pos + neg)) if (pos + neg) else 0.0

    return n, pos, neg, unl, unk, miss, unique_videos, int(round(mean_per_video)), pos_frac, max_per_video, has_joints, has_motion, has_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Base folder containing split subfolders.")
    args = ap.parse_args()

    candidates = ["train", "val", "test", "unsplit", "test_unlabeled"]
    splits = [d for d in candidates if os.path.isdir(os.path.join(args.root, d))]
    if not splits:
        raise SystemExit(f"[ERR] No split folders found under: {args.root}")

    print(f"[scan] {args.root}")
    for s in splits:
        (n, p, n0, unl, u, m, vids, mean_v, pos_frac, max_v, hj, hm, hk) = count_split(os.path.join(args.root, s))
        print(
            f"{s:15s} files={n:6d}  pos={p:6d}  neg={n0:6d}  unl={unl:6d}  "
            f"unk={u:6d}  missing_y={m:6d}  videos={vids:5d}  "
            f"win/video≈{mean_v:6d}  max/video={max_v:5d}  pos_frac={pos_frac:5.2f}  "
            f"schema(joints/motion/mask)={hj:6d}/{hm:6d}/{hk:6d}"
        )


if __name__ == "__main__":
    main()
