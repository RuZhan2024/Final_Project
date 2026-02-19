#!/usr/bin/env python3
"""
LE2i labels + fall spans builder (rewrite).

Updates:
- Verbose warnings on annotation read/parse errors.
- Final summary stats always printed (npz_total, matched_labels, missing/skipped).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
from typing import Dict, List, Optional, Tuple

_VIDEO_RE = re.compile(r"__video__\(?(\d+)\)?_", re.IGNORECASE)

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

    for pat in _CANDIDATE_PATTERNS:
        p = os.path.join(ann_dir, pat.format(i=vid_idx))
        if os.path.isfile(p):
            return p

    for p in glob.glob(os.path.join(ann_dir, "*.txt")):
        name = os.path.basename(p).lower()
        if re.search(rf"video\s*[(_ ]?{vid_idx}[)_ ]?\.txt$", name):
            return p
    return None


def parse_span_txt(txt_path: str, *, verbose: bool = False) -> List[List[int]]:
    """
    Extract fall span(s) from LE2i annotation file.
    Returns [[start, end)] in raw-frame units.
    """
    try:
        lines = pathlib.Path(txt_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        if verbose:
            print(f"[warn] failed to read annotation txt: {txt_path} ({e})")
        return []

    singles: List[int] = []
    for line in lines[:50]:
        ints = [int(x) for x in re.findall(r"-?\d+", line)]
        if not ints:
            continue
        if len(ints) == 1:
            singles.append(ints[0])
            if len(singles) >= 2:
                s, e = singles[0], singles[1]
                break
        elif len(ints) == 2:
            s, e = ints[0], ints[1]
            break
        else:
            continue
    else:
        nums: List[int] = []
        for line in lines[:10]:
            nums.extend(int(x) for x in re.findall(r"-?\d+", line))
            if len(nums) >= 2:
                break
        if len(nums) < 2:
            return []
        s, e = nums[0], nums[1]

    if s < 0:
        s = 0
    if e <= s:
        return []
    return [[int(s), int(e)]]


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
            e2 += 1

        if clamp_len is not None:
            s2 = max(0, min(s2, clamp_len))
            e2 = max(0, min(e2, clamp_len))

        if e2 > s2:
            out.append([s2, e2])

    out.sort(key=lambda x: (x[0], x[1]))
    merged: List[List[int]] = []
    for s, e in out:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged


def maybe_get_seq_len(npz_dir: str, stem: str) -> Optional[int]:
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
    ap.add_argument("--end_inclusive", action="store_true")
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--clamp_to_npz_len", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    stems = list_npz_stems(args.npz_dir)
    if not stems:
        raise SystemExit(f"[ERR] No npz under {args.npz_dir}")

    labels: Dict[str, str] = {}
    spans_out: Dict[str, List[List[int]]] = {}

    labeled, unlabeled = 0, 0
    fall_count = 0
    skipped_bad_stem = 0

    for stem in stems:
        scene, vid = stem_to_scene_and_vid(stem)
        if scene is None or vid is None:
            skipped_bad_stem += 1
            continue

        ann = find_annotation_file(args.raw_root, scene, vid)
        if ann is None:
            unlabeled += 1
            if args.include_unannotated_as_adl:
                labels[stem] = "adl"
            continue

        labeled += 1
        raw_spans = parse_span_txt(ann, verbose=args.verbose)

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
    print(f"[info] stems scanned: {len(stems)}  annotated_included={labeled}  without_annotation={unlabeled}  bad_stem_skipped={skipped_bad_stem}")
    if not args.include_unannotated_as_adl:
        print("[info] sequences without annotations were excluded (use --include_unannotated_as_adl to include as ADL)")

    # Required verification stats
    total_npz = len(stems)
    matched_labels = len(labels)
    missing = total_npz - matched_labels
    print(f"[summary] npz_total={total_npz}  matched_labels={matched_labels}  missing_or_skipped={missing}")

if __name__ == "__main__":
    main()
