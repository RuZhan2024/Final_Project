#!/usr/bin/env python3
import os, glob, json, pathlib, re, argparse

def stem_to_label(stem: str) -> str:
    s = stem.lower()
    # split on -, _, ., __ etc.
    toks = re.split(r"[._\-]+", s)
    if "fall" in toks or "__fall__" in s:
        return "fall"
    if "adl" in toks or "__adl__" in s:
        return "adl"
    # default conservative
    return "adl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default="data/interim/urfd/pose_npz")
    ap.add_argument("--out",     default="configs/labels/urfd_auto.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.npz_dir, "**", "*.npz"), recursive=True))
    if not files:
        raise SystemExit(f"No NPZ under {args.npz_dir}")

    labels = {}
    fall = adl = 0
    for p in files:
        stem = pathlib.Path(p).stem
        lab = stem_to_label(stem)
        labels[stem] = lab
        fall += (lab == "fall")
        adl  += (lab == "adl")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f: json.dump(labels, f, indent=2)
    print(f"[OK] wrote {len(labels)} → {args.out}  (fall={fall}, adl={adl})")

if __name__ == "__main__":
    main()
