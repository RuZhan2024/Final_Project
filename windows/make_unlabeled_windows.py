
"""
Create fixed-length windows for *unlabeled* sequences (e.g., LE2I Office/Lecture).

Writes NPZ windows at: {out_dir}/{subset}/{stem}__w{start:06d}_{end:06d}.npz

Each window NPZ contains (schema aligned with make_windows.py):
  - xy       : [W, 33, 2]
  - conf     : [W, 33]
  - y        : -1  (explicitly unlabeled)
  - label    : -1  (kept for backward compatibility)
  - start    : start frame index (inclusive)
  - end      : end frame index (inclusive)
  - video_id : stem (string)
  - stem     : stem (string, kept for backward compatibility)
  - fps      : fps of the source sequence (float32)
"""

import os
import glob
import argparse
import pathlib
import sys
import numpy as np


def read_stems(stems_txt):
    stems = []
    with open(stems_txt, "r", encoding="utf-8") as f:
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


def window_sequence(npz_path, W, stride, fps_default):
    with np.load(npz_path, allow_pickle=False) as z:
        xy = np.nan_to_num(
            z["xy"].astype(np.float32),
            nan=0.0, posinf=0.0, neginf=0.0
        )
        conf = np.nan_to_num(
            z["conf"].astype(np.float32),
            nan=0.0, posinf=0.0, neginf=0.0
        )
        fps = float(z["fps"]) if "fps" in z.files else float(fps_default)

    T = int(xy.shape[0])
    if T < W:
        return xy, conf, fps, []

    starts = list(range(0, T - W + 1, stride))
    return xy, conf, fps, starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--stems_txt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--fps_default", type=float, default=30.0,
                    help="Fallback FPS if source NPZ lacks fps (default: 30).")
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
    too_short = 0

    for stem in stems:
        p = idx.get(stem)
        if p is None:
            missing.append(stem)
            continue

        xy, conf, fps, starts = window_sequence(p, args.W, args.stride, args.fps_default)
        if not starts:
            too_short += 1
            continue

        for s in starts:
            e = s + args.W - 1  # inclusive
            xy_w = xy[s:s + args.W]
            conf_w = conf[s:s + args.W]

            out_p = os.path.join(out_base, f"{stem}__w{s:06d}_{e:06d}.npz")
            np.savez_compressed(
                out_p,
                xy=xy_w,
                conf=conf_w,
                y=np.int64(-1),
                label=np.int32(-1),          # backward compat
                start=np.int64(s),
                end=np.int64(e),
                video_id=str(stem),
                stem=str(stem),              # backward compat
                fps=np.float32(fps),
            )
            total_win += 1

    print(f"[OK] wrote {total_win} windows → {out_base}")
    if too_short:
        print(f"[WARN] {too_short} sequences shorter than W={args.W} (skipped).")
    if missing:
        print(f"[WARN] {len(missing)} stems not found in {args.npz_dir}. Examples: {missing[:5]}")


if __name__ == "__main__":
    main()
