#!/usr/bin/env python3
"""
Create fixed-length windows for *unlabeled* sequences (e.g., LE2i Office/Lecture).

Inputs
------
--npz_dir     : root of extracted sequence NPZs (xy, conf) from extract_2d.py
--stems_txt   : text file with one NPZ stem per line (e.g., Office/Lecture stems)
--out_dir     : base output dir for windows; this script writes into out_dir/<subset>
--W           : window length in frames (e.g., 48)
--stride      : stride in frames between windows (e.g., 12)
--subset      : name of subfolder to write into (default: test_unlabeled)

Output
------
NPZ windows at: {out_dir}/{subset}/{stem}_t{start:06d}.npz
Each NPZ contains:
  - xy   : [W, 33, 2]
  - conf : [W, 33]
  - stem : original sequence stem (string)
  - start: start frame index of the window (int)
  - label: -1  (explicitly mark unlabeled)
"""

import os, glob, argparse, pathlib, sys
import numpy as np

def read_stems(stems_txt):
    stems = []
    with open(stems_txt, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): 
                continue
            stems.append(s)
    return stems

def index_npz(npz_root):
    idx = {}
    files = glob.glob(os.path.join(npz_root, "**", "*.npz"), recursive=True)
    for p in files:
        idx[pathlib.Path(p).stem] = p
    return idx

def window_sequence(npz_path, W, stride):
    z = np.load(npz_path, allow_pickle=False)
    xy   = np.nan_to_num(z["xy"].astype(np.float32))
    conf = np.nan_to_num(z["conf"].astype(np.float32))
    T = xy.shape[0]
    starts = list(range(0, max(T - W + 1, 0), stride))
    return xy, conf, starts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--stems_txt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--subset", default="test_unlabeled")
    args = ap.parse_args()

    stems = read_stems(args.stems_txt)
    if not stems:
        sys.exit(f"[ERR] No stems found in {args.stems_txt}")

    idx = index_npz(args.npz_dir)
    out_base = os.path.join(args.out_dir, args.subset)
    os.makedirs(out_base, exist_ok=True)

    total_win = 0
    missing = []
    for stem in stems:
        p = idx.get(stem)
        if p is None:
            missing.append(stem)
            continue
        xy, conf, starts = window_sequence(p, args.W, args.stride)
        for s in starts:
            xy_w   = xy[s:s+args.W]
            conf_w = conf[s:s+args.W]
            out_p = os.path.join(out_base, f"{stem}_t{s:06d}.npz")
            np.savez_compressed(
                out_p,
                xy=xy_w,
                conf=conf_w,
                stem=np.array(stem),
                start=np.int32(s),
                label=np.int32(-1),
            )
            total_win += 1

    print(f"[OK] wrote {total_win} windows → {out_base}")
    if missing:
        print(f"[WARN] {len(missing)} stems not found in {args.npz_dir}. Examples: {missing[:5]}")

if __name__ == "__main__":
    main()
