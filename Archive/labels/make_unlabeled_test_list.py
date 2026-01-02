#!/usr/bin/env python3
"""
Utility: write a list of NPZ stems that match scene keywords (used for unlabeled LE2i scenes).

Example
-------
python labels/make_unlabeled_test_list.py \
  --npz_dir data/interim/le2i/pose_npz \
  --out configs/splits/le2i_unlabeled.txt \
  --scenes Office "Lecture room"

Notes
-----
- Matching is case-insensitive and ignores spaces/underscores differences.
- This script does NOT check whether stems already exist in labels.json; it's just a picker.
"""

import argparse
import glob
import os
import pathlib
import re
from typing import List

def norm(s: str) -> str:
    return re.sub(r"[\s_]+", "_", s.lower())

def list_npz_files(npz_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="Root of NPZs (e.g., data/interim/le2i/pose_npz)")
    ap.add_argument("--out", required=True, help="Output txt with one stem per line")
    ap.add_argument("--scenes", nargs="+", required=True, help='Scene names to match, e.g. Office "Lecture room"')
    args = ap.parse_args()

    files = list_npz_files(args.npz_dir)
    if not files:
        raise SystemExit(f"[ERR] No NPZ under {args.npz_dir}")

    scene_keys = [norm(s) for s in args.scenes]
    picked = []
    for p in files:
        pkey = norm(p)
        if any(sk in pkey for sk in scene_keys):
            picked.append(pathlib.Path(p).stem)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(picked) + ("\n" if picked else ""))

    print(f"[OK] wrote {len(picked)} stems → {args.out}")
    if picked[:10]:
        print("[sample]", ", ".join(picked[:10]))

if __name__ == "__main__":
    main()
