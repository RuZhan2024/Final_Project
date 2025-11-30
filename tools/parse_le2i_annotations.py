#!/usr/bin/env python3
"""
Parse LE2i Annotation_files to produce:
  - configs/labels/le2i.json        (video-level labels: 'adl' or 'fall')
  - configs/labels/le2i_spans.json  (frame spans for falls per video)

Only videos whose scenes have Annotation_files are labeled; others (e.g., Office,
Lecture room) are intentionally left out so you can use them as unlabeled test data.
"""

import os, re, glob, json, pathlib, argparse

# NPZ stems look like: Coffee_room_01__Videos__video__10_
def stem_to_scene_and_index(stem: str):
    toks = stem.split("__")
    scene = toks[0] if toks else None
    m = re.search(r"__video__\(?(\d+)\)?_", stem, re.IGNORECASE)
    vid = int(m.group(1)) if m else None
    return scene, vid

# Annotation filename variants we try
CANDIDATE_PATTERNS = [
    "video ({i}).txt", "Video ({i}).txt",
    "video({i}).txt",  "Video({i}).txt",
    "video_{i}.txt",   "Video_{i}.txt",
    "video {i}.txt",   "Video {i}.txt",
]

def parse_fall_txt(path):
    """Return a list of [start,end] pairs (ints). Most LE2i files contain 1 pair."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        return []
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    spans = []
    for i in range(0, len(nums) - 1, 2):
        s, e = nums[i], nums[i+1]
        if e > s >= 0:
            spans.append([s, e])
            break  # take the first span; remove if you want multiple
    return spans

def find_annotation(raw_root, scene, vid_idx):
    ann_dir = os.path.join(raw_root, scene, "Annotation_files")
    if not os.path.isdir(ann_dir):
        return None
    for pat in CANDIDATE_PATTERNS:
        p = os.path.join(ann_dir, pat.format(i=vid_idx))
        if os.path.isfile(p):
            return p
    for p in glob.glob(os.path.join(ann_dir, "*.txt")):
        name = os.path.basename(p).lower()
        if re.search(rf"video\s*[(_ ]?{vid_idx}[)_ ]?\.txt$", name):
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root",  default="data/raw/LE2i")
    ap.add_argument("--npz_dir",   default="data/interim/le2i/pose_npz")
    ap.add_argument("--out_labels", default="configs/labels/le2i.json")
    ap.add_argument("--out_spans",  default="configs/labels/le2i_spans.json")
    args = ap.parse_args()

    npzs = sorted(glob.glob(os.path.join(args.npz_dir, "**", "*.npz"), recursive=True))
    if not npzs:
        raise SystemExit(f"No NPZs under {args.npz_dir}")

    labels, spans = {}, {}
    for p in npzs:
        stem = pathlib.Path(p).stem
        scene, vid = stem_to_scene_and_index(stem)
        if scene is None or vid is None:
            continue  # skip malformed stems

        ann = find_annotation(args.raw_root, scene, vid)
        if ann is None:
            # No annotations for this scene/video → leave unlabeled on purpose
            continue

        sps = parse_fall_txt(ann)
        if sps:
            labels[stem] = "fall"
            spans[stem]  = sps
        else:
            labels[stem] = "adl"

    os.makedirs(os.path.dirname(args.out_labels) or ".", exist_ok=True)
    with open(args.out_labels, "w") as f: json.dump(labels, f, indent=2)
    with open(args.out_spans,  "w") as f: json.dump(spans,  f, indent=2)

    print(f"[OK] labels → {args.out_labels} ({len(labels)} labeled videos)")
    print(f"[OK] spans  → {args.out_spans}  ({len(spans)} with fall spans)")

if __name__ == "__main__":
    main()
