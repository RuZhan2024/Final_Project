
"""
check_windows.py

Scan a windows directory (train/val/test/unsplit/test_unlabeled) and report:
- total files
- positives (y==1)
- negatives (y==0)
- unlabeled/unknown (y not in {0,1}, e.g. -1)
- missing label key (no y field)

Usage:
  python windows/check_windows.py --root data/processed/le2i/windows_W48_S12
"""

import os
import glob
import argparse
import numpy as np


def count_split(path: str):
    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    pos = neg = unk = miss = 0

    for p in files:
        try:
            with np.load(p, allow_pickle=False) as z:
                if "y" not in z.files:
                    miss += 1
                    continue
                y = int(np.array(z["y"]).reshape(()))
        except Exception:
            # Corrupt or unreadable NPZ counts as missing label
            miss += 1
            continue

        if y == 1:
            pos += 1
        elif y == 0:
            neg += 1
        else:
            unk += 1

    return len(files), pos, neg, unk, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="Base folder containing train/val/test or other split subfolders.",
    )
    args = ap.parse_args()

    candidates = ["train", "val", "test", "unsplit", "test_unlabeled"]
    splits = [d for d in candidates if os.path.isdir(os.path.join(args.root, d))]
    if not splits:
        raise SystemExit(f"[ERR] No split folders found under: {args.root}")

    print(f"[scan] {args.root}")
    for s in splits:
        n, p, n0, u, m = count_split(os.path.join(args.root, s))
        print(
            f"{s:15s} files={n:6d}  pos={p:6d}  neg={n0:6d}  "
            f"unk(else)={u:6d}  missing_y={m:6d}"
        )


if __name__ == "__main__":
    main()
