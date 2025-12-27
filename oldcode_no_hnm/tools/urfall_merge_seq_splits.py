#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".txt"}

def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    os.symlink(os.path.relpath(src, dst.parent), dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="data/raw/UR_Fall_seq")
    ap.add_argument("--out_root", required=True, help="data/raw/UR_Fall_clips")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    total_links = 0
    clips = set()

    for split in ["train", "valid", "test"]:
        sroot = in_root / split
        if not sroot.exists():
            continue
        for clip_dir in sroot.iterdir():
            if not clip_dir.is_dir():
                continue
            clips.add(clip_dir.name)
            out_clip = out_root / clip_dir.name
            for f in clip_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMG_EXT:
                    link(f, out_clip / f.name)
                    total_links += 1

    print(f"[OK] merged clips={len(clips)} links={total_links} → {out_root}")

if __name__ == "__main__":
    main()

# python3 tools/urfall_merge_seq_splits.py \
#   --in_root data/raw/UR_Fall_seq \
#   --out_root data/raw/UR_Fall_clips
