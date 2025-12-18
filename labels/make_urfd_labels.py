
"""
URFD – build video-level labels from NPZ stems.

Assumes pose NPZs have been created by extract_2d_from_images.py, with filenames
containing 'fall' / 'adl' (either in parent directory names or the stem itself).

Reads:
  --npz_dir : root of URFD pose NPZs

Writes:
  configs/labels/urfd.json  with entries:
    { "some_stem": "fall", "other_stem": "adl", ... }
"""

import os, glob, json, pathlib, re, argparse


def stem_to_label(stem: str) -> str:
    """
    Infer 'fall' vs 'adl' from the stem.

    Strategy:
      - lower-case the stem
      - split on separators (., _, -)
      - if any token == 'fall'  → 'fall'
      - if any token == 'adl'   → 'adl'
      - default to 'adl' (conservative)
    """
    s = stem.lower()
    toks = re.split(r"[._\-]+", s)
    if "fall" in toks or "__fall__" in s:
        return "fall"
    if "adl" in toks or "__adl__" in s:
        return "adl"
    return "adl"  # safe default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", default="data/interim/urfd/pose_npz",
                    help="URFD pose NPZs from extract_2d_from_images.py")
    ap.add_argument("--out",     default="configs/labels/urfd.json",
                    help="Where to write URFD labels JSON.")
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
    with open(args.out, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"[OK] wrote {len(labels)} labels → {args.out}  (fall={fall}, adl={adl})")


if __name__ == "__main__":
    main()
