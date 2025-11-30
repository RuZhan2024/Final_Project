#!/usr/bin/env python3
"""
Check window folders for label balance and missing labels.
Default path assumes LE2i 48x stride-12 windows.
"""

import os, glob, argparse, numpy as np

def count_split(path):
    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    pos = neg = unk = miss = 0
    for p in files:
        z = np.load(p, allow_pickle=False)
        if "y" not in z.files:
            miss += 1; continue
        y = int(np.array(z["y"]).reshape(()))
        if y == 1: pos += 1
        elif y == 0: neg += 1
        else: unk += 1
    return len(files), pos, neg, unk, miss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/le2i/windows_W48_S12",
                    help="Base folder containing train/val/test/unsplit/test_unlabeled")
    args = ap.parse_args()

    splits = [d for d in ["train","val","test","unsplit","test_unlabeled"]
              if os.path.isdir(os.path.join(args.root, d))]
    if not splits:
        raise SystemExit(f"No split folders found under {args.root}")

    print(f"[scan] {args.root}")
    for s in splits:
        n, p, n0, u, m = count_split(os.path.join(args.root, s))
        print(f"{s:15s} files={n:6d}  pos={p:6d}  neg={n0:6d}  unk(-1)={u:6d}  missing_y={m:6d}")

if __name__ == "__main__":
    main()
