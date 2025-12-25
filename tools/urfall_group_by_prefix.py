#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# fall-01-cam0-rgb-081_png  ->  fall-01-cam0-rgb
RE = re.compile(r"^(.*)-(\d{1,6})_png$", re.IGNORECASE)

def clip_id(stem: str) -> str:
    m = RE.match(stem)
    if m:
        return m.group(1)
    # fallback: strip last "-123" if present
    m2 = re.match(r"^(.*)-(\d{1,6})$", stem)
    return m2.group(1) if m2 else stem

def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    os.symlink(os.path.relpath(src, dst.parent), dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="data/raw/UR_Fall")
    ap.add_argument("--out_root", required=True, help="data/raw/UR_Fall_seq")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ["train","valid","test"]:
        d = in_root / split
        if not d.exists():
            continue

        imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]
        print(f"[{split}] images={len(imgs)}")

        for img in imgs:
            stem = img.stem
            cid = clip_id(stem)
            out_dir = out_root / split / cid
            link(img, out_dir / img.name)

            txt = d / f"{stem}.txt"
            if txt.exists():
                link(txt, out_dir / txt.name)

    print(f"[OK] grouped → {out_root}")

if __name__ == "__main__":
    main()


# python3 tools/urfall_group_by_prefix.py \
#   --in_root data/raw/UR_Fall \
#   --out_root data/raw/UR_Fall_seq
