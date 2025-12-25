#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def strip_rf(stem: str) -> str:
    # "base.rf.hash" -> "base"
    return stem.split(".rf.")[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data/raw/UR_Fall")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    splits = ["train", "valid", "test"]

    for split in splits:
        d = root / split
        if not d.exists():
            continue

        files = [p for p in d.iterdir() if p.is_file() and (p.suffix.lower() in IMG_EXT or p.suffix.lower()==".txt")]

        # Build plan and detect collisions
        plan = []
        seen = {}
        for p in files:
            new_stem = strip_rf(p.stem)
            new_name = new_stem + p.suffix.lower()
            new_path = p.with_name(new_name)

            if new_path.exists() and new_path != p:
                raise SystemExit(f"[ERR] target already exists: {new_path}")

            key = str(new_path)
            if key in seen and seen[key] != str(p):
                raise SystemExit(f"[ERR] collision: {p} and {seen[key]} -> {new_path}")
            seen[key] = str(p)

            if new_path != p:
                plan.append((p, new_path))

        print(f"[{split}] rename candidates: {len(plan)}")

        # Execute
        for src, dst in plan:
            print(f"  {src.name} -> {dst.name}")
            if not args.dry_run:
                src.rename(dst)

    print("[OK] done")

if __name__ == "__main__":
    main()
