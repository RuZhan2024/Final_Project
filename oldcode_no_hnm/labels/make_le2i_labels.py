#!/usr/bin/env python3
"""
LE2i labels + fall spans builder (rewrite).

Reads
-----
- Cleaned pose sequences (.npz) under --npz_dir (default: data/interim/le2i/pose_npz)
- LE2i annotation txt files under --raw_root (default: data/raw/LE2i)

Writes
------
- --out_labels : JSON mapping {stem: "adl"|"fall"} (only for sequences with annotations by default)
- --out_spans  : JSON mapping {stem: [[start, stop], ...]} with half-open spans [start, stop)

Why this rewrite?
-----------------
1) Keeps frame indices stable (we do NOT drop frames in preprocess) so spans remain aligned.
2) Adds robust filename matching for many LE2i annotation filename variants.
3) Adds clamping/scaling options in case your pose extraction sampled frames.

Notes
-----
- By default, scenes/videos with no annotation file are *excluded* from outputs so you can
  treat them as unlabeled test data. Use --include_unannotated_as_adl to include them as ADL.

Span conventions
----------------
- Spans are half-open [start, stop) in pose-frame index units.
- If your pose extraction used frame_stride > 1 (e.g., took every 2nd frame), set --frame_stride=2
  so spans are mapped from raw-frame indices to pose-frame indices: pose_idx = raw_idx // frame_stride.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
from typing import Dict, List, Optional, Tuple

# MediaPipe Pose doesn't matter here; we only use stems.

# NPZ stems often look like: Coffee_room_01__Videos__video__10_
_VIDEO_RE = re.compile(r"__video__\(?(\d+)\)?_", re.IGNORECASE)

# Annotation filename variants we try (LE2i is inconsistent across releases)
_CANDIDATE_PATTERNS = [
    "video ({i}).txt", "Video ({i}).txt",
    "video({i}).txt",  "Video({i}).txt",
    "video_{i}.txt",   "Video_{i}.txt",
    "video {i}.txt",   "Video {i}.txt",
]

def list_npz_stems(npz_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))
    return [pathlib.Path(p).stem for p in files]

def stem_to_scene_and_vid(stem: str) -> Tuple[Optional[str], Optional[int]]:
    toks = stem.split("__")
    scene = toks[0] if toks else None
    m = _VIDEO_RE.search(stem)
    vid = int(m.group(1)) if m else None
    return scene, vid

def find_annotation_file(raw_root: str, scene: str, vid_idx: int) -> Optional[str]:
    ann_dir = os.path.join(raw_root, scene, "Annotation_files")
    if not os.path.isdir(ann_dir):
        return None

    # 1) deterministic patterns
    for pat in _CANDIDATE_PATTERNS:
        p = os.path.join(ann_dir, pat.format(i=vid_idx))
        if os.path.isfile(p):
            return p

    # 2) fuzzy fallback
    for p in glob.glob(os.path.join(ann_dir, "*.txt")):
        name = os.path.basename(p).lower()
        if re.search(rf"video\s*[(_ ]?{vid_idx}[)_ ]?\.txt$", name):
            return p
    return None

def parse_span_txt(txt_path: str) -> List[List[int]]:
    """
    Return spans from annotation text.
    Most LE2i txt contain 2 ints: start end (inclusive or exclusive depends on release).
    We treat as [start, stop) by default and will normalise later via --end_inclusive.
    """
    try:
        text = pathlib.Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    spans: List[List[int]] = []
    # Accept multiple pairs, but keep them ordered & non-empty
    for i in range(0, len(nums) - 1, 2):
        s, e = nums[i], nums[i + 1]
        if s < 0:
            s = 0
        if e > s:
            spans.append([s, e])
    return spans

def normalise_spans(
    spans: List[List[int]],
    *,
    end_inclusive: bool,
    frame_stride: int,
    clamp_len: Optional[int],
) -> List[List[int]]:
    out: List[List[int]] = []
    stride = max(1, int(frame_stride))

    for s, e in spans:
        s2 = int(s) // stride
        e2 = int(e) // stride
        if end_inclusive:
            e2 += 1  # inclusive -> exclusive

        if clamp_len is not None:
            s2 = max(0, min(s2, clamp_len))
            e2 = max(0, min(e2, clamp_len))

        if e2 > s2:
            out.append([s2, e2])

    # merge overlaps
    out.sort(key=lambda x: (x[0], x[1]))
    merged: List[List[int]] = []
    for s, e in out:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged

def maybe_get_seq_len(npz_dir: str, stem: str) -> Optional[int]:
    # Best-effort: open the matching npz and read xy length
    path = None
    for p in glob.glob(os.path.join(npz_dir, "**", stem + ".npz"), recursive=True):
        path = p
        break
    if not path:
        return None
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as z:
            if "xy" in z:
                return int(z["xy"].shape[0])
    except Exception:
        return None
    return None

def main():
    ap = argparse.ArgumentParser(description="Build LE2i labels + fall spans.")
    ap.add_argument("--raw_root", default="data/raw/LE2i", help="LE2i raw root containing scenes/*/Annotation_files")
    ap.add_argument("--npz_dir", default="data/interim/le2i/pose_npz", help="Cleaned pose npz dir for LE2i")
    ap.add_argument("--out_labels", default="configs/labels/le2i.json")
    ap.add_argument("--out_spans", default="configs/labels/le2i_spans.json")
    ap.add_argument("--include_unannotated_as_adl", action="store_true",
                    help="Include sequences without annotation txt as ADL labels.")
    ap.add_argument("--end_inclusive", action="store_true",
                    help="Treat annotation end as inclusive, i.e. convert (start,end) -> [start,end+1).")
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="If pose extraction sampled every k frames from the raw video, set k here.")
    ap.add_argument("--clamp_to_npz_len", action="store_true",
                    help="Clamp spans to the sequence length read from corresponding pose npz (best-effort).")
    args = ap.parse_args()

    stems = list_npz_stems(args.npz_dir)
    if not stems:
        raise SystemExit(f"[ERR] No npz under {args.npz_dir}")

    labels: Dict[str, str] = {}
    spans_out: Dict[str, List[List[int]]] = {}

    labeled, unlabeled = 0, 0
    fall_count = 0

    for stem in stems:
        scene, vid = stem_to_scene_and_vid(stem)
        if scene is None or vid is None:
            continue

        ann = find_annotation_file(args.raw_root, scene, vid)
        if ann is None:
            unlabeled += 1
            if args.include_unannotated_as_adl:
                labels[stem] = "adl"
            continue

        # We have an annotation file -> include in labels
        labeled += 1
        raw_spans = parse_span_txt(ann)

        clamp_len = maybe_get_seq_len(args.npz_dir, stem) if args.clamp_to_npz_len else None
        spans = normalise_spans(
            raw_spans,
            end_inclusive=bool(args.end_inclusive),
            frame_stride=int(args.frame_stride),
            clamp_len=clamp_len,
        )

        if spans:
            labels[stem] = "fall"
            spans_out[stem] = spans
            fall_count += 1
        else:
            labels[stem] = "adl"

    os.makedirs(os.path.dirname(args.out_labels) or ".", exist_ok=True)
    with open(args.out_labels, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    os.makedirs(os.path.dirname(args.out_spans) or ".", exist_ok=True)
    with open(args.out_spans, "w", encoding="utf-8") as f:
        json.dump(spans_out, f, indent=2)

    adl_count = sum(1 for v in labels.values() if v == "adl")
    print(f"[OK] wrote labels  → {args.out_labels}  (total={len(labels)}, fall={fall_count}, adl={adl_count})")
    print(f"[OK] wrote spans   → {args.out_spans}   (videos_with_spans={len(spans_out)})")
    print(f"[info] stems scanned: {len(stems)}  annotated_included={labeled}  without_annotation={unlabeled}")
    if not args.include_unannotated_as_adl:
        print("[info] sequences without annotations were excluded (use --include_unannotated_as_adl to include as ADL)")

if __name__ == "__main__":
    main()
